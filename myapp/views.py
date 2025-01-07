from django.shortcuts import render
from django.http import JsonResponse
from binance.client import Client
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import time
import gc
from django.views.decorators.csrf import csrf_exempt
from .services.technical_indicators import calculate_technical_indicators
from .services.data_preprocessing import DataPreprocessingService
from .services.data_combination import DataCombinationService
from .services.model_interface import ModelInterface
from django.views.decorators.http import require_http_methods
import logging

logger = logging.getLogger(__name__)

# Create your views here.

def home(request):
    return render(request, 'myapp/home.html')

def data_collection(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            selected_tickers = data.get('tickers', [])
            interval = data.get('interval')
            days = int(data.get('days', 30))
            
            # Binance API 설정
            api_key = os.getenv('BINANCE_API_KEY', '')
            api_secret = os.getenv('BINANCE_API_SECRET', '')
            client = Client(api_key, api_secret)
            
            # 데이터 저장 경로 설정
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(BASE_DIR, 'data', 'raw_data')
            os.makedirs(data_dir, exist_ok=True)
            
            print(f"Data directory: {data_dir}")  # 디버깅을 위한 경로 출력
            
            # 데이터 수집 기간 설정
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # API 요청 제한 관리를 위한 변수
            request_count = 0
            last_request_time = time.time()
            requests_per_minute = 1200  # Binance API 제한
            chunk_size = timedelta(days=1)  # 1일 단위로 청크 분할
            
            collected_data = []
            data_info = {}  # 각 티커별 데이터 정보를 저장할 딕셔너리
            
            for ticker in selected_tickers:
                current_start = start_date
                all_data = pd.DataFrame()
                
                while current_start < end_date:
                    # API 요청 속도 제한 관리
                    request_count += 1
                    current_time = time.time()
                    
                    if request_count >= requests_per_minute:
                        elapsed_time = current_time - last_request_time
                        if elapsed_time < 60:
                            sleep_time = 60 - elapsed_time
                            time.sleep(sleep_time)
                        request_count = 0
                        last_request_time = time.time()
                    elif request_count % 10 == 0:
                        time.sleep(0.5)
                    
                    # 청크 단위로 데이터 수집
                    current_end = min(current_start + chunk_size, end_date)
                    
                    try:
                        klines = client.get_historical_klines(
                            ticker, 
                            interval,
                            int(current_start.timestamp() * 1000),
                            int(current_end.timestamp() * 1000)
                        )
                        
                        if klines:
                            chunk_df = pd.DataFrame(klines, columns=[
                                'datetime', 'open', 'high', 'low', 'close', 'volume',
                                'close_time', 'quote_volume', 'trade_count',
                                'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
                            ])
                            
                            # 필요한 컬럼만 선택
                            chunk_df = chunk_df[['datetime', 'open', 'high', 'low', 'close', 'volume',
                                               'quote_volume', 'trade_count', 'taker_buy_volume',
                                               'taker_buy_quote_volume']]
                            
                            # datetime 변환 및 형식 지정
                            chunk_df['datetime'] = pd.to_datetime(chunk_df['datetime'], unit='ms')
                            chunk_df['datetime'] = chunk_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                            
                            # 숫자 데이터 형변환 및 소수점 자리수 조정
                            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                                            'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
                            for col in numeric_columns:
                                chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')
                                chunk_df[col] = chunk_df[col].round(4)
                            
                            # trade_count는 정수로 변환
                            chunk_df['trade_count'] = pd.to_numeric(chunk_df['trade_count'], errors='coerce').round(0)
                            
                            all_data = pd.concat([all_data, chunk_df], ignore_index=True)
                            
                    except Exception as e:
                        print(f"Error collecting data for {ticker} at {current_start}: {str(e)}")
                        time.sleep(1)  # 에러 발생 시 잠시 대기
                    
                    current_start = current_end
                    gc.collect()  # 메모리 정리
                
                if not all_data.empty:
                    # 중복 제거 및 정렬
                    all_data = all_data.drop_duplicates(subset=['datetime'])
                    all_data = all_data.sort_values('datetime')
                    
                    # CSV 저장
                    csv_filename = f'{ticker}_raw_data.csv'
                    csv_path = os.path.join(data_dir, csv_filename)
                    all_data.to_csv(csv_path, index=False, float_format='%.4f')
                    
                    print(f"Saving CSV to: {csv_path}")  # 디버깅을 위한 저장 경로 출력
                    
                    # 데이터 정보 저장
                    data_info[ticker] = len(all_data)
                    
                    # 최근 3개 데이터만 화면에 표시용으로 저장
                    latest_data = all_data.tail(3).to_dict('records')
                    collected_data.extend(latest_data)
                
                del all_data
                gc.collect()
            
            # 전체 데이터 건수 계산
            total_rows = sum(data_info.values())
            
            return JsonResponse({
                'status': 'success',
                'data': collected_data,
                'data_info': data_info,
                'total_rows': total_rows,
                'save_path': data_dir  # 저장 경로 정보 추가
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    else:
        try:
            # ticker_list.txt 읽기
            ticker_file = os.path.join('data', 'ticker_list.txt')
            with open(ticker_file, 'r') as f:
                tickers = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            tickers = []
        
        intervals = [
            ('1m', '1분'),
            ('5m', '5분'),
            ('30m', '30분'),
            ('1h', '1시간'),
            ('1d', '1일'),
            ('1w', '1주'),
        ]
        
        context = {
            'tickers': tickers,
            'intervals': intervals,
        }
        
        return render(request, 'myapp/data_collection.html', context)

def technical_indicators(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            selected_indicators = data.get('indicators', [])
            
            # 데이터 저장 경로 설정
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            raw_data_dir = os.path.join(BASE_DIR, 'data', 'raw_data')
            processed_data_dir = os.path.join(BASE_DIR, 'data', 'processed_data')
            os.makedirs(processed_data_dir, exist_ok=True)
            
            # raw_data.csv 파일들 처리
            total_rows = 0
            processed_data = []
            
            for file in os.listdir(raw_data_dir):
                if file.endswith('_raw_data.csv'):
                    ticker = file.replace('_raw_data.csv', '')
                    df = pd.read_csv(os.path.join(raw_data_dir, file))
                    
                    # 기술적 지표 계산
                    df_with_indicators = calculate_technical_indicators(df)
                    
                    # 결과 저장
                    output_file = os.path.join(processed_data_dir, f'{ticker}_feature_data.csv')
                    df_with_indicators.to_csv(output_file, index=False)
                    
                    total_rows += len(df_with_indicators)
                    
                    # 최근 3개 데이터만 화면에 표시
                    latest_data = df_with_indicators.tail(3).to_dict('records')
                    processed_data.extend(latest_data)
            
            return JsonResponse({
                'status': 'success',
                'data': processed_data,
                'data_info': {
                    'total_rows': total_rows,
                    'save_path': processed_data_dir
                }
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    return render(request, 'myapp/technical_indicators.html')

def technical_indicators_view(request):
    if request.method == 'POST':
        try:
            # JSON 데이터 파싱
            data = json.loads(request.body)
            selected_indicators = data.get('indicators', {})
            
            # 입력/출력 디렉토리 설정 (절대 경로 사용)
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            raw_data_dir = os.path.join(BASE_DIR, 'data', 'raw_data')
            feature_data_dir = os.path.join(BASE_DIR, 'data', 'feature_data')
            
            # 출력 디렉토리 생성
            os.makedirs(feature_data_dir, exist_ok=True)
            
            print(f"Raw data directory: {raw_data_dir}")  # 디버깅용
            print(f"Feature data directory: {feature_data_dir}")  # 디버깅용
            print(f"Selected indicators: {selected_indicators}")  # 디버깅용

            processed_files = 0
            latest_preview = None

            # raw_data 디렉토리의 모든 CSV 파일 처리
            for root, _, files in os.walk(raw_data_dir):
                for file in files:
                    if file.endswith('_raw_data.csv'):
                        try:
                            # 입력 파일 경로
                            input_path = os.path.join(root, file)
                            print(f"Processing file: {input_path}")  # 디버깅용
                            
                            # 출력 파일 경로 설정
                            relative_path = os.path.relpath(root, raw_data_dir)
                            output_dir = os.path.join(feature_data_dir, relative_path)
                            os.makedirs(output_dir, exist_ok=True)
                            
                            output_file = file.replace('_raw_data.csv', '_feature_data.csv')
                            output_path = os.path.join(output_dir, output_file)
                            print(f"Output path: {output_path}")  # 디버깅용

                            # 데이터 처리
                            df = pd.read_csv(input_path)
                            if df.empty:
                                print(f"Warning: Empty dataframe for {file}")
                                continue
                                
                            processed_df = calculate_technical_indicators(df, selected_indicators=selected_indicators)
                            
                            # 결과 저장
                            processed_df.to_csv(output_path, index=False)
                            print(f"Saved to: {output_path}")  # 디버깅용
                            
                            processed_files += 1
                            
                            # 마지막 파일의 미리보기 데이터 저장
                            latest_preview = processed_df.tail(3)

                        except Exception as e:
                            print(f"Error processing {file}: {str(e)}")  # 디버깅용
                            continue

            if processed_files == 0:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No files were processed. Check if raw data files exist.'
                })

            return JsonResponse({
                'status': 'success',
                'processed_files': processed_files,
                'preview_data': latest_preview.to_dict('records') if latest_preview is not None else None,
                'message': f'Successfully processed {processed_files} files.'
            })

        except Exception as e:
            print(f"Error: {str(e)}")  # 디버깅용
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)

    # GET 요청 시 초기 데이터 전달
    context = {
        'ma_periods': [5, 10, 20, 60, 120, 240],
        'rsi_periods': [5, 10, 20, 60, 120],
        'roc_periods': [10, 20, 40],
        'stochastic_settings': [
            {'value': '5-3-3', 'label': '5-3-3'},
            {'value': '10-6-6', 'label': '10-6-6'},
            {'value': '20-12-12', 'label': '20-12-12'},
        ]
    }
    return render(request, 'myapp/technical_indicators.html', context)

@csrf_exempt
def get_tickers(request):
    ticker_file_path = os.path.join('data', 'ticker_list.txt')
    if os.path.exists(ticker_file_path):
        with open(ticker_file_path, 'r') as f:
            tickers = [line.strip() for line in f.readlines()]
        return JsonResponse(tickers, safe=False)
    return JsonResponse([], safe=False)

@csrf_exempt
def data_preprocessing_view(request):
    if request.method == 'GET':
        return render(request, 'myapp/data_preprocessing.html')
    
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # 설정값 가져오기
            scaler_type = data.get('scaler_type', 'minmax')
            sequence_length = int(data.get('sequence_length', 300))
            train_ratio = float(data.get('train_ratio', 70)) / 100
            val_ratio = float(data.get('val_ratio', 15)) / 100
            
            # 입력/출력 디렉토리 설정
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            input_dir = os.path.join(base_dir, 'data', 'feature_data')
            output_dir = os.path.join(base_dir, 'data', 'processed_data')
            
            # 데이터 전처리 서비스 실행
            service = DataPreprocessingService(input_dir, output_dir)
            result = service.process_data(
                scaler_type=scaler_type,
                sequence_length=sequence_length,
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )
            
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })

@csrf_exempt
def data_combination_view(request):
    if request.method == 'GET':
        return render(request, 'myapp/data_combination.html')
    
    elif request.method == 'POST':
        try:
            # 입력/출력 디렉토리 설정
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            input_dir = os.path.join(base_dir, 'data', 'processed_data')
            output_dir = os.path.join(base_dir, 'data', 'processed_data')
            
            # 데이터 결합 서비스 실행
            service = DataCombinationService(input_dir, output_dir)
            result = service.combine_data()
            
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })

@csrf_exempt
def create_model(request):
    """모델 생성 API"""
    if request.method == 'POST':
        try:
            # JSON 데이터 파싱
            data = json.loads(request.body)
            print("Received data:", data)  # 디버깅용 로그
            
            # 설정 데이터 추출
            config = {
                'window_size': int(data.get('window_size', 100)),
                'feature_count': int(data.get('feature_count', 10)),
                'patterns': data.get('patterns', ['double_bottom', 'double_top']),
                'risk_management': {
                    'stop_loss': float(data.get('stop_loss', 0.02)),
                    'take_profit': float(data.get('take_profit', 0.04))
                }
            }
            
            print("Config:", config)  # 디버깅용 로그
            
            # 모델 생성
            interface = ModelInterface()
            model = interface.create_model_from_strategy(config)
            
            # 모델 저장 경로 생성
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, 'model')
            os.makedirs(model_dir, exist_ok=True)
            
            # 현재 시간을 포함한 모델 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f'model_{timestamp}'
            model_path = os.path.join(model_dir, model_name)
            
            # 모델 저장
            interface.save_model(model_path)
            
            return JsonResponse({
                'status': 'success',
                'message': '모델이 성공적으로 생성되었습니다.',
                'model_path': model_path
            })
            
        except json.JSONDecodeError as e:
            print("JSON Decode Error:", str(e))  # 디버깅용 로그
            return JsonResponse({
                'status': 'error',
                'message': f'잘못된 JSON 형식입니다: {str(e)}'
            }, status=400)
            
        except Exception as e:
            print("Error:", str(e))  # 디버깅용 로그
            return JsonResponse({
                'status': 'error',
                'message': f'모델 생성 중 오류가 발생했습니다: {str(e)}'
            }, status=500)
    
    return JsonResponse({
        'status': 'error',
        'message': '잘못된 요청 방식입니다.'
    }, status=400)

@csrf_exempt
@require_http_methods(["POST"])
def train_model(request):
    """모델 학습 API 엔드포인트"""
    try:
        # 요청 데이터 파싱
        data = json.loads(request.body)
        data_path = data.get('data_path')
        
        logger.info(f"Received training data: {data}")
        
        # ModelInterface 인스턴스 가져오기
        model_interface = ModelInterface()
        
        # 모델 학습 수행
        result = model_interface.train_model(data_dir=data_path)
        
        if result['status'] == 'success':
            # 학습 결과 반환
            return JsonResponse({
                'status': 'success',
                'message': '모델 학습이 완료되었습니다.',
                'history': result['history']
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': result['message']
            }, status=500)
        
    except Exception as e:
        logger.error(f"모델 학습 중 오류 발생: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'모델 학습 중 오류가 발생했습니다: {str(e)}'
        }, status=500)

@csrf_exempt
def model_development_view(request):
    """모델 개발 페이지를 렌더링합니다."""
    context = {
        'available_patterns': [
            {'value': 'double_bottom', 'label': '쌍바닥'},
            {'value': 'double_top', 'label': '쌍봉'},
            {'value': 'head_and_shoulders', 'label': '헤드앤숄더'},
            {'value': 'inverse_head_and_shoulders', 'label': '역헤드앤숄더'},
        ],
        'window_sizes': [20, 50, 100, 200],
        'feature_counts': [10, 20, 30, 50]
    }
    return render(request, 'myapp/model_development.html', context)

def get_data_info(request):
    """학습 데이터의 shape 정보를 반환하는 API"""
    try:
        model_interface = ModelInterface()
        logger.info("ModelInterface 인스턴스 생성됨")
        
        # 데이터 디렉토리 확인
        data_dir = model_interface.default_data_dir
        logger.info(f"데이터 디렉토리 확인: {data_dir}")
        
        if not os.path.exists(data_dir):
            logger.error(f"데이터 디렉토리가 존재하지 않음: {data_dir}")
            return JsonResponse({
                'error': f'데이터 디렉토리를 찾을 수 없습니다: {data_dir}'
            }, status=404)
            
        # 필요한 파일들 확인
        required_files = ['combined_X_train.npy', 'combined_y_train.npy', 'feature.pkl', 'scaler.pkl']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
        
        if missing_files:
            logger.error(f"필요한 파일이 없음: {missing_files}")
            return JsonResponse({
                'error': f'다음 파일들이 없습니다: {", ".join(missing_files)}'
            }, status=404)
        
        # 데이터 로드
        X_train, _ = model_interface.load_data()
        logger.info(f"학습 데이터 shape - X_train: {X_train.shape}")
        
        response_data = {
            'window_size': int(X_train.shape[1]),  # 시간 스텝 수
            'feature_count': int(X_train.shape[2])  # 특성 수
        }
        logger.info(f"반환할 데이터: {response_data}")
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"데이터 정보 로드 중 오류 발생: {str(e)}", exc_info=True)
        return JsonResponse({
            'error': f'데이터 로드 중 오류가 발생했습니다: {str(e)}'
        }, status=500)
