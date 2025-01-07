import os
import numpy as np
import logging
import shutil
import pickle

logger = logging.getLogger(__name__)

class DataCombinationService:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.output_dir = os.path.join(base_dir, 'data', 'learning_data')
        os.makedirs(self.output_dir, exist_ok=True)

    def get_array_shape(self, filename):
        """파일을 메모리에 로드하지 않고 npy 파일의 shape 정보 읽기"""
        with open(filename, 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        return shape

    def combine_data(self):
        """데이터 결합 실행"""
        try:
            # 결합할 데이터 파일들
            data_types = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
            combined_data = {dtype: [] for dtype in data_types}
            file_info = {dtype: [] for dtype in data_types}
            
            # feature.pkl과 scaler.pkl을 저장할 변수 초기화
            feature_infos = []
            first_scaler = None
            
            # 모든 하위 폴더 탐색
            for ticker_folder in os.listdir(self.input_dir):
                folder_path = os.path.join(self.input_dir, ticker_folder)
                if not os.path.isdir(folder_path):
                    continue

                # feature.pkl과 scaler.pkl 처리
                feature_path = os.path.join(folder_path, 'feature.pkl')
                scaler_path = os.path.join(folder_path, 'scaler.pkl')
                
                # 모든 feature 정보 수집
                if os.path.exists(feature_path):
                    with open(feature_path, 'rb') as f:
                        feature_info = pickle.load(f)
                        feature_infos.append({
                            'ticker': ticker_folder,
                            'info': feature_info
                        })
                
                if os.path.exists(scaler_path) and first_scaler is None:
                    with open(scaler_path, 'rb') as f:
                        first_scaler = pickle.load(f)

                # 각 데이터 타입별로 파일 결합
                for dtype in data_types:
                    file_path = os.path.join(folder_path, f'{dtype}.npy')
                    if os.path.exists(file_path):
                        try:
                            # 파일 shape 정보 저장
                            shape = self.get_array_shape(file_path)
                            file_info[dtype].append({
                                'ticker': ticker_folder,
                                'shape': shape
                            })
                            # 데이터 로드 및 결합 준비
                            data = np.load(file_path, mmap_mode='r')
                            combined_data[dtype].append(data)
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {str(e)}")
                            continue

            # feature 정보 검증
            if not feature_infos:
                raise ValueError("No feature information found in any ticker folder")
            
            # 모든 티커의 feature 수가 동일한지 확인
            feature_counts = [len(info['info']['feature_columns']) for info in feature_infos]
            if len(set(feature_counts)) > 1:
                logger.error("Feature count mismatch between tickers:")
                for info in feature_infos:
                    logger.error(f"- {info['ticker']}: {len(info['info']['feature_columns'])} features")
                raise ValueError("Feature count mismatch between tickers")
            
            # 첫 번째 feature 정보 사용
            first_feature_info = feature_infos[0]['info']
            logger.info(f"Using feature info from {feature_infos[0]['ticker']}")
            logger.info(f"Feature columns: {first_feature_info['feature_columns']}")

            # 데이터 결합 및 저장
            results = {}
            for dtype in data_types:
                if combined_data[dtype]:
                    try:
                        # shape 검증
                        shapes = [data.shape for data in combined_data[dtype]]
                        if len(set(shape[1:] for shape in shapes)) > 1:
                            logger.error(f"Shape mismatch in {dtype}:")
                            for info in file_info[dtype]:
                                logger.error(f"- {info['ticker']}: {info['shape']}")
                            raise ValueError(f"Shape mismatch in {dtype}")
                        
                        combined = np.concatenate(combined_data[dtype], axis=0)
                        output_path = os.path.join(self.output_dir, f'combined_{dtype}.npy')
                        np.save(output_path, combined)
                        results[dtype] = {
                            'shape': combined.shape,
                            'files': file_info[dtype]
                        }
                        logger.info(f"Combined {dtype} shape: {combined.shape}")
                    except Exception as e:
                        logger.error(f"Error combining {dtype}: {str(e)}")
                        continue

            # feature.pkl과 scaler.pkl 저장
            if first_feature_info is not None:
                with open(os.path.join(self.output_dir, 'feature.pkl'), 'wb') as f:
                    pickle.dump(first_feature_info, f)
                logger.info("Saved feature.pkl")
            
            if first_scaler is not None:
                with open(os.path.join(self.output_dir, 'scaler.pkl'), 'wb') as f:
                    pickle.dump(first_scaler, f)
                logger.info("Saved scaler.pkl")

            return {
                'status': 'success',
                'results': results,
                'output_dir': self.output_dir,
                'feature_info': {
                    'count': len(first_feature_info['feature_columns']),
                    'columns': first_feature_info['feature_columns']
                }
            }

        except Exception as e:
            logger.error(f"Error during data combination: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            } 