import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging
import pickle

logger = logging.getLogger(__name__)

class DataPreprocessingService:
    # 스케일링 제외할 컬럼 정의를 클래스 변수로 이동
    exclude_from_scaling = [
        'pct_change_1min', 'pct_change_5min', 'pct_change_10min', 'volatility',
        'Support Breakout 10d', 'Support Breakout 20d', 'Support Breakout 60d',
        'Support Breakout 120d', 'Resistance Breakout 20d', 'MACD',
        'MACD_signal', 'HV', 'CMF'
    ]

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def get_scaler(self, scaler_type):
        scalers = {
            'minmax': MinMaxScaler,
            'standard': StandardScaler,
            'robust': RobustScaler
        }
        return scalers.get(scaler_type, MinMaxScaler)()

    def prepare_lstm_data(self, df, sequence_length, scaler_type='minmax'):
        """LSTM 데이터 준비"""
        try:
            # 스케일링 제용할 컬럼 선택
            feature_columns = [col for col in df.columns if col not in ['datetime', 'Ticker']]
            scaling_columns = [col for col in feature_columns if col not in self.exclude_from_scaling]

            # 스케일링 적용
            scaler = self.get_scaler(scaler_type)
            scaled_values = scaler.fit_transform(df[scaling_columns])

            # 최종 데이터 준비
            final_data = np.zeros((len(df), len(feature_columns)), dtype=np.float32)
            
            # 스케일링된 데이터 할당
            for i, col in enumerate(scaling_columns):
                col_idx = feature_columns.index(col)
                final_data[:, col_idx] = scaled_values[:, i]

            # 스케일링 제외 데이터 할당
            for col in self.exclude_from_scaling:
                if col in feature_columns:
                    col_idx = feature_columns.index(col)
                    final_data[:, col_idx] = df[col].values

            # LSTM 시퀀스 생성
            X, y = [], []
            for i in range(len(final_data) - sequence_length):
                X.append(final_data[i:i + sequence_length])
                y.append(final_data[i + sequence_length])

            return np.array(X), np.array(y), scaler, feature_columns, scaling_columns

        except Exception as e:
            logger.error(f"LSTM 데이터 준비 중 오류 발생: {str(e)}")
            raise

    def process_data(self, scaler_type='minmax', sequence_length=300, train_ratio=0.7, val_ratio=0.15):
        """데이터 전처리 실행"""
        processed_files = 0
        preview_data = None

        try:
            # 입력 디렉토리의 모든 CSV 파일 처리
            for filename in os.listdir(self.input_dir):
                if not filename.endswith('_feature_data.csv'):
                    continue

                ticker = filename.split('_feature_data.csv')[0]
                input_path = os.path.join(self.input_dir, filename)
                
                # CSV 파일 읽기
                df = pd.read_csv(input_path)
                
                # LSTM 데이터 준비
                X, y, scaler, feature_columns, scaling_columns = self.prepare_lstm_data(
                    df, sequence_length, scaler_type
                )

                # 데이터 분할
                test_ratio = 1 - train_ratio - val_ratio
                
                # 먼저 train+val과 test 분할
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_ratio, shuffle=False
                )
                
                # train과 validation 분할
                val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_ratio_adjusted, shuffle=False
                )

                # 결과 저장
                output_dir = os.path.join(self.output_dir, ticker)
                os.makedirs(output_dir, exist_ok=True)

                # NPY 파일 저장
                np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
                np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
                np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
                np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
                np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
                np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

                # 스케일링된 데이터 CSV 저장
                scaled_df = pd.DataFrame(y, columns=feature_columns)
                scaled_df.to_csv(os.path.join(output_dir, 'scaled_data.csv'), index=False)

                # feature 정보 저장
                feature_info = {
                    'feature_columns': feature_columns,
                    'scaling_columns': scaling_columns,
                    'exclude_from_scaling': self.exclude_from_scaling
                }
                with open(os.path.join(output_dir, 'feature.pkl'), 'wb') as f:
                    pickle.dump(feature_info, f)

                # scaler 저장
                with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
                    pickle.dump(scaler, f)

                # 미리보기 데이터 준비 (마지막 처리된 파일의 최근 3개 행)
                preview_data = scaled_df.tail(3)
                
                processed_files += 1

            return {
                'status': 'success',
                'processed_files': processed_files,
                'preview_data': preview_data.to_dict('records') if preview_data is not None else None
            }

        except Exception as e:
            logger.error(f"데이터 처리 중 오류 발생: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            } 