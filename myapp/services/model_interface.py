import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# GPU 메모리 할당 설정
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

logger = logging.getLogger(__name__)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 나머지 코드는 그대로...

class ModelInterface:
    def __init__(self):
        self.model = None
        self.input_shape = None
        self.output_shape = None
        self.feature_info = None
        self.scaler = None
        self.strategy_config = None
        
        # 프로젝트 루트 디렉토리 찾기 (myproject 폴더)
        current_file = os.path.abspath(__file__)
        myapp_dir = os.path.dirname(os.path.dirname(current_file))
        project_root = os.path.dirname(myapp_dir)
        
        # 기본 데이터 디렉토리 설정
        self.default_data_dir = os.path.join(project_root, 'data', 'learning_data')
        logger.info(f"기본 데이터 디렉토리 설정: {self.default_data_dir}")

    def create_model_from_strategy(self, strategy_config):
        """전략 설정을 기반으로 모델 설계"""
        try:
            logger.info(f"전략 설정으로 모델 설계 시작: {strategy_config}")
            self.strategy_config = strategy_config

            # 학습 데이터에서 shape 정보 추출
            try:
                X_train, y_train = self.load_data()
                window_size = X_train.shape[1]  # 시간 스텝 수
                feature_count = X_train.shape[2]  # 특성 수
                logger.info("학습 데이터 shape 정보:")
                logger.info(f"- 전체 샘플 수: {X_train.shape[0]}")
                logger.info(f"- 시퀀스 길이(window_size): {window_size}")
                logger.info(f"- 특성 수(feature_count): {feature_count}")
                logger.info(f"- y_train shape: {y_train.shape}")
            except Exception as e:
                logger.error(f"학습 데이터 shape 추출 실패: {str(e)}")
                raise

            self.input_shape = (window_size, feature_count)
            self.output_shape = feature_count
            logger.info(f"모델 shape 설정 - 입력: {self.input_shape}, 출력: {self.output_shape}")

            # 모델 생성
            self.create_model()

            # 초기 모델 저장
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                model_dir = os.path.join(base_dir, 'model')
                os.makedirs(model_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # 초기 모델 저장 (.keras 형식)
                model_path = os.path.join(model_dir, f'model_{timestamp}.keras')
                self.model.save(model_path)
                logger.info(f"초기 모델 저장 완료: {model_path}")
                
                # 초기 설정 저장
                config = {
                    'input_shape': self.input_shape,
                    'output_shape': self.output_shape,
                    'strategy_config': self.strategy_config,
                    'feature_info': self.feature_info
                }
                config_path = os.path.join(model_dir, f'model_{timestamp}_config.pkl')
                with open(config_path, 'wb') as f:
                    pickle.dump(config, f)
                logger.info(f"초기 모델 설정 저장 완료: {config_path}")
                logger.info(f"저장된 설정: {config}")
                
            except Exception as e:
                logger.warning(f"초기 모델 저장 실패: {str(e)}")

            return {
                'status': 'success',
                'message': '모델이 성공적으로 설계되었습니다.',
                'model_info': {
                    'input_shape': self.input_shape,
                    'output_shape': self.output_shape,
                    'strategy': strategy_config,
                    'data_info': {
                        'total_samples': X_train.shape[0],
                        'window_size': window_size,
                        'feature_count': feature_count
                    }
                }
            }

        except Exception as e:
            logger.error(f"모델 설계 중 오류 발생: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'모델 설계 중 오류가 발생했습니다: {str(e)}'
            }

    def create_model(self):
        """데이터 shape에 맞는 동적 모델 생성"""
        try:
            if self.input_shape is None or self.output_shape is None:
                raise ValueError("모델 생성 전에 shape 정보가 필요합니다.")

            # 입력 레이어
            inputs = layers.Input(shape=self.input_shape)
            
            # LSTM 레이어 (동적 유닛 수 계산)
            lstm_units = min(256, self.input_shape[-1] * 4)  # feature 수의 4배, 최대 256
            x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
            x = layers.Dropout(0.2)(x)
            
            # 두 번째 LSTM 레이어
            x = layers.LSTM(lstm_units // 2, return_sequences=False)(x)
            x = layers.Dropout(0.2)(x)
            
            # Dense 레이어들 (동적 유닛 수 계산)
            dense_units = min(512, lstm_units * 2)
            x = layers.Dense(dense_units, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.Dense(dense_units // 2, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            
            # 출력 레이어
            outputs = layers.Dense(self.output_shape)(x)
            
            # 모델 컴파일
            self.model = models.Model(inputs=inputs, outputs=outputs)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info(f"모델 생성 완료 - 입력 shape: {self.input_shape}, 출력 shape: {self.output_shape}")
            return self.model
            
        except Exception as e:
            logger.error(f"모델 생성 중 오류 발생: {str(e)}")
            raise

    def load_data(self, data_dir=None):
        """학습 데이터 로드 및 shape 확인"""
        try:
            # 데이터 디렉토리가 지정되지 않은 경우 기본값 사용
            if data_dir is None:
                data_dir = self.default_data_dir
            else:
                # 상대 경로를 절대 경로로 변환
                if not os.path.isabs(data_dir):
                    current_file = os.path.abspath(__file__)
                    myapp_dir = os.path.dirname(os.path.dirname(current_file))
                    project_root = os.path.dirname(myapp_dir)
                    data_dir = os.path.join(project_root, data_dir)

            # 데이터 디렉토리 생성
            os.makedirs(data_dir, exist_ok=True)
            logger.info(f"데이터 디렉토리 확인/생성: {data_dir}")

            if not os.path.exists(data_dir):
                raise ValueError(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")

            # 데이터 파일 경로
            x_train_path = os.path.join(data_dir, 'combined_X_train.npy')
            y_train_path = os.path.join(data_dir, 'combined_y_train.npy')
            feature_path = os.path.join(data_dir, 'feature.pkl')
            scaler_path = os.path.join(data_dir, 'scaler.pkl')

            # 파일 존재 확인 및 자세한 로깅
            missing_files = []
            for path in [x_train_path, y_train_path, feature_path, scaler_path]:
                if not os.path.exists(path):
                    missing_files.append(os.path.basename(path))
            
            if missing_files:
                raise ValueError(f"다음 파일들이 {data_dir}에 없습니다: {', '.join(missing_files)}")

            # 데이터 로드
            X_train = np.load(x_train_path)
            y_train = np.load(y_train_path)
            
            # feature 정보와 scaler 로드
            with open(feature_path, 'rb') as f:
                self.feature_info = pickle.load(f)
                logger.info(f"Feature 정보: {self.feature_info}")
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info(f"데이터 로드 완료:")
            logger.info(f"- X_train shape: {X_train.shape} (샘플 수, 시퀀스 길이, 특성 수)")
            logger.info(f"- y_train shape: {y_train.shape} (샘플 수, 특성 수)")
            logger.info(f"- 특성 컬럼 수: {len(self.feature_info['feature_columns'])}")
            logger.info(f"- 스케일링된 컬럼 수: {len(self.feature_info['scaling_columns'])}")
            logger.info(f"- 스케일링 제외 컬럼 수: {len(self.feature_info['exclude_from_scaling'])}")
            
            return X_train, y_train
            
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}", exc_info=True)
            raise

    def find_latest_model(self):
        """가장 최근에 설계된 모델 찾기"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base_dir, 'model')
            
            if not os.path.exists(model_dir):
                logger.warning(f"모델 디렉토리를 찾을 수 없습니다: {model_dir}")
                return None
                
            # 모델 파일 찾기
            model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.keras')]
            if not model_files:
                logger.warning("설계된 모델을 찾을 수 없습니다.")
                return None
                
            # 가장 최근 모델 선택
            latest_model = max(model_files)
            model_path = os.path.join(model_dir, latest_model)
            logger.info(f"최근 설계된 모델 발견: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"최근 모델 검색 중 오류 발생: {str(e)}")
            return None

    # train_model 메소드 수정
    def train_model(self, data_dir=None, epochs=5, batch_size=32, validation_split=0.2):
        """모델 학습 및 히스토리 저장"""
        try:
            # 데이터 로드
            X_train, y_train = self.load_data(data_dir)
            logger.info(f"학습 데이터 shape - X: {X_train.shape}, y: {y_train.shape}")
            
            # base_dir과 model_dir 설정
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base_dir, 'model')
            os.makedirs(model_dir, exist_ok=True)
            
            # 모델이 없으면 최근 설계된 모델 로드
            if self.model is None:
                model_files = [f for f in os.listdir(model_dir) if f.startswith('model_') and f.endswith('.keras')]
                if not model_files:
                    raise ValueError("설계된 모델을 찾을 수 없습니다.")
                
                latest_model = max(model_files)
                model_path = os.path.join(model_dir, latest_model)
                config_path = os.path.join(model_dir, latest_model.replace('.keras', '_config.pkl'))
                
                self.model = models.load_model(model_path)
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)
                    self.input_shape = config['input_shape']
                    self.output_shape = config['output_shape']
                    self.strategy_config = config['strategy_config']
                    self.feature_info = config['feature_info']
                
                logger.info(f"최근 설계된 모델 로드 완료: {model_path}")

            # 타임스탬프 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 학습 로그 파일 경로
            log_dir = os.path.join(model_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f'training_log_{timestamp}.csv')
            
            # 콜백 정의
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(model_dir, f'checkpoint_model_{timestamp}_'+'{epoch:02d}.keras'),
                    monitor='val_loss',
                    save_best_only=True
                ),
                tf.keras.callbacks.CSVLogger(
                    log_path,
                    separator=',',
                    append=False
                )
            ]
            
            # 모델 학습
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # 학습된 최종 모델 저장
            best_model_path = os.path.join(model_dir, f'best_model_{timestamp}.keras')
            self.model.save(best_model_path)
            
            # 학습 히스토리를 딕셔너리로 변환
            history_dict = {
                'loss': history.history['loss'],
                'mae': history.history['mae'],
                'val_loss': history.history['val_loss'],
                'val_mae': history.history['val_mae']
            }
            
            # 설정 및 히스토리 저장
            config = {
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'strategy_config': self.strategy_config,
                'feature_info': self.feature_info,
                'training_params': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'validation_split': validation_split
                },
                'training_history': history_dict,
                'timestamp': timestamp
            }
            
            best_config_path = os.path.join(model_dir, f'best_model_config_{timestamp}.pkl')
            with open(best_config_path, 'wb') as f:
                pickle.dump(config, f)
            
            # 학습 결과 텍스트 파일 생성
            results_path = os.path.join(log_dir, f'training_results_{timestamp}.txt')
            with open(results_path, 'w', encoding='utf-8') as f:
                f.write(f"모델 학습 결과 요약\n")
                f.write(f"====================\n")
                f.write(f"학습 시작 시간: {timestamp}\n")
                f.write(f"데이터 정보:\n")
                f.write(f"- 학습 데이터 크기: {X_train.shape}\n")
                f.write(f"- 특성 수: {self.input_shape[-1]}\n")
                f.write(f"\n학습 파라미터:\n")
                f.write(f"- 에포크: {epochs}\n")
                f.write(f"- 배치 크기: {batch_size}\n")
                f.write(f"- 검증 분할: {validation_split}\n")
                f.write(f"\n최종 성능:\n")
                f.write(f"- 손실 (Loss): {history_dict['loss'][-1]:.6f}\n")
                f.write(f"- MAE: {history_dict['mae'][-1]:.6f}\n")
                f.write(f"- 검증 손실: {history_dict['val_loss'][-1]:.6f}\n")
                f.write(f"- 검증 MAE: {history_dict['val_mae'][-1]:.6f}\n")
            
            logger.info(f"학습된 최종 모델 저장 완료: {best_model_path}")
            logger.info(f"학습 설정 및 히스토리 저장 완료: {best_config_path}")
            logger.info(f"학습 결과 저장 완료: {results_path}")
            
            return {
                'status': 'success',
                'message': '모델 학습이 완료되었습니다.',
                'history': history_dict,
                'model_path': best_model_path,
                'model_info': {
                    'input_shape': self.input_shape,
                    'output_shape': self.output_shape,
                    'total_epochs': len(history_dict['loss']),
                    'final_loss': history_dict['loss'][-1],
                    'final_val_loss': history_dict['val_loss'][-1],
                    'final_mae': history_dict['mae'][-1],
                    'final_val_mae': history_dict['val_mae'][-1]
                },
                'log_paths': {
                    'training_log': log_path,
                    'results_summary': results_path
                }
            }
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'모델 학습 중 오류가 발생했습니다: {str(e)}'
            }
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'모델 학습 중 오류가 발생했습니다: {str(e)}'
            }


    def save_model(self, save_path):
        """모델 저장"""
        try:
            if self.model is None:
                raise ValueError("저장할 모델이 없습니다.")
            
            # 저장 디렉토리 생성
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            
            # 기본 경로에서 확장자 제거
            base_path = save_path.replace('.keras', '')
            
            # 모델 저장 (.keras 형식)
            model_path = f"{base_path}.keras"
            self.model.save(model_path)
            logger.info(f"모델 저장 완료: {model_path}")
            
            # 설정 저장 (.pkl 형식)
            config = {
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'feature_info': self.feature_info,
                'strategy_config': self.strategy_config
            }
            
            config_path = f"{base_path}_config.pkl"
            with open(config_path, 'wb') as f:
                pickle.dump(config, f)
            logger.info(f"설정 저장 완료: {config_path}")
            
        except Exception as e:
            logger.error(f"모델 저장 중 오류 발생: {str(e)}")
            raise

    def load_model(self, model_path):
        """모델 로드"""
        try:
            # 기본 경로에서 확장자 제거
            base_path = model_path.replace('.keras', '')
            
            # 모델 로드
            model_path_full = f"{base_path}.keras"
            self.model = models.load_model(model_path_full)
            logger.info(f"모델 로드 완료: {model_path_full}")
            
            # 설정 로드
            config_path = f"{base_path}_config.pkl"
            if os.path.exists(config_path):
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)
                
                self.input_shape = config['input_shape']
                self.output_shape = config['output_shape']
                self.feature_info = config.get('feature_info')
                self.strategy_config = config.get('strategy_config')
                logger.info(f"설정 로드 완료: {config_path}")
            else:
                logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise

    def predict(self, X):
        """예측 수행"""
        try:
            if self.model is None:
                raise ValueError("예측을 위한 모델이 로드되지 않았습니다.")
            
            # 입력 데이터 shape 확인 및 조정
            if len(X.shape) == 2:  # 단일 샘플인 경우
                X = np.expand_dims(X, axis=0)
            
            if X.shape[1:] != self.input_shape:
                raise ValueError(f"입력 데이터 shape가 모델의 입력 shape와 다릅니다. "
                               f"예상: {self.input_shape}, 실제: {X.shape[1:]}")
            
            # 예측 수행
            predictions = self.model.predict(X)
            
            return predictions
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            raise 

    def visualize_training_history(self, history_dict, save_dir):
        """학습 히스토리 시각화 및 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 스타일 설정
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 1. Loss와 MAE 그래프
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history_dict['loss'], label='Training Loss')
        ax1.plot(history_dict['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(history_dict['mae'], label='Training MAE')
        ax2.plot(history_dict['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE Over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'training_history_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

    def save_performance_metrics(self, history_dict, save_path):
        """모델 성능 메트릭 저장"""
        final_epoch = len(history_dict['loss'])
        best_epoch = np.argmin(history_dict['val_loss']) + 1
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("모델 성능 분석 보고서\n")
            f.write("====================\n\n")
            
            f.write("1. 학습 개요\n")
            f.write(f"- 총 에포크: {final_epoch}\n")
            f.write(f"- 최적 에포크: {best_epoch}\n")
            f.write(f"- 학습 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("2. 최종 성능 지표\n")
            f.write(f"- 최종 학습 손실: {history_dict['loss'][-1]:.6f}\n")
            f.write(f"- 최종 검증 손실: {history_dict['val_loss'][-1]:.6f}\n")
            f.write(f"- 최종 학습 MAE: {history_dict['mae'][-1]:.6f}\n")
            f.write(f"- 최종 검증 MAE: {history_dict['val_mae'][-1]:.6f}\n\n")
            
            f.write("3. 최적 성능 지표\n")
            f.write(f"- 최적 검증 손실: {min(history_dict['val_loss']):.6f} (에포크 {best_epoch})\n")
            f.write(f"- 최적 검증 MAE: {min(history_dict['val_mae']):.6f}\n\n")
            
            f.write("4. 성능 개선 분석\n")
            initial_val_loss = history_dict['val_loss'][0]
            final_val_loss = history_dict['val_loss'][-1]
            improvement = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100
            
            f.write(f"- 초기 검증 손실: {initial_val_loss:.6f}\n")
            f.write(f"- 최종 검증 손실: {final_val_loss:.6f}\n")
            f.write(f"- 성능 개선율: {improvement:.2f}%\n\n")
            
            f.write("5. 과적합 분석\n")
            train_val_gap = history_dict['loss'][-1] - history_dict['val_loss'][-1]
            f.write(f"- 학습-검증 격차: {abs(train_val_gap):.6f}\n")
            if train_val_gap > 0.01:
                f.write("- 경고: 과적합 징후가 감지됨\n")
            else:
                f.write("- 정상: 과적합 징후 없음\n")          