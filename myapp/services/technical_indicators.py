import numpy as np
import pandas as pd

def calculate_technical_indicators(df, selected_indicators=None, prefix='', min_change_rate=0.005):
    """
    선택된 기술적 지표만 계산합니다.
    :param df: 원본 데이터프레임
    :param selected_indicators: 선택된 지표들의 딕셔너리
    :param prefix: 컬럼명 접두사
    :param min_change_rate: 최소 변화율
    """
    if selected_indicators is None:
        selected_indicators = {}

    result_df = df.copy()
    epsilon = 1e-10  # 0으로 나누기 방지를 위한 작은 값

    price_cols = {
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    }

    # 1. 추세 지표 (Trend Indicators)
    # 이동평균선
    if 'ma' in selected_indicators:
        for period in selected_indicators['ma']:
            period = int(period)
            result_df[f'{prefix}MA_{period}'] = result_df[price_cols['close']].rolling(window=period).mean().round(4)

    # MACD
    if 'macd' in selected_indicators:
        exp1 = result_df[price_cols['close']].ewm(span=12, adjust=False).mean()
        exp2 = result_df[price_cols['close']].ewm(span=26, adjust=False).mean()
        result_df[f'{prefix}MACD'] = (exp1 - exp2).round(4)
        result_df[f'{prefix}MACD_signal'] = result_df[f'{prefix}MACD'].ewm(span=9, adjust=False).mean().round(4)
        result_df[f'{prefix}MACD_hist'] = (result_df[f'{prefix}MACD'] - result_df[f'{prefix}MACD_signal']).round(4)

    # DMI
    if 'dmi' in selected_indicators:
        high_delta = result_df[price_cols['high']].diff()
        low_delta = result_df[price_cols['low']].diff()
        
        plus_dm = (high_delta > 0) & (high_delta > low_delta.abs()) * high_delta
        minus_dm = (low_delta < 0) & (low_delta.abs() > high_delta) * low_delta.abs()
        
        tr = pd.DataFrame({
            'hl': result_df[price_cols['high']] - result_df[price_cols['low']],
            'hc': abs(result_df[price_cols['high']] - result_df[price_cols['close']].shift(1)),
            'lc': abs(result_df[price_cols['low']] - result_df[price_cols['close']].shift(1))
        }).max(axis=1)
        
        period = 14
        tr14 = tr.rolling(window=period).sum()
        plus_di14 = (100 * (plus_dm.rolling(window=period).sum() / tr14)).round(4)
        minus_di14 = (100 * (minus_dm.rolling(window=period).sum() / tr14)).round(4)
        
        result_df[f'{prefix}DMI_plus_di'] = plus_di14
        result_df[f'{prefix}DMI_minus_di'] = minus_di14
        result_df[f'{prefix}DMI_adx'] = (100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)).round(4)

    # 2. 모멘텀 지표 (Momentum Indicators)
    # RSI
    if 'rsi' in selected_indicators:
        for period in selected_indicators['rsi']:
            period = int(period)
            delta = result_df[price_cols['close']].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, epsilon)
            result_df[f'{prefix}RSI_{period}'] = (100 - (100 / (1 + rs))).round(4)

    # 스토캐스틱
    if 'stochastic' in selected_indicators:
        for setting in selected_indicators['stochastic']:
            k_period, d_period, sd_period = map(int, setting.split('-'))
            low_min = result_df[price_cols['low']].rolling(window=k_period).min()
            high_max = result_df[price_cols['high']].rolling(window=k_period).max()
            
            price_range = high_max - low_min
            price_range = price_range.replace(0, epsilon)
            
            fast_k = ((result_df[price_cols['close']] - low_min) / price_range * 100)
            fast_k = fast_k.replace([np.inf, -np.inf], 50).round(4)
            
            fast_d = fast_k.rolling(window=d_period).mean().round(4)
            slow_d = fast_d.rolling(window=sd_period).mean().round(4)
            
            result_df[f'{prefix}Stoch{k_period}_%K'] = fast_k
            result_df[f'{prefix}Stoch{k_period}_%D'] = fast_d
            result_df[f'{prefix}Stoch{k_period}_Slow_%D'] = slow_d

    # ROC
    if 'roc' in selected_indicators:
        for period in selected_indicators['roc']:
            period = int(period)
            result_df[f'{prefix}ROC_{period}'] = (
                (result_df[price_cols['close']] - result_df[price_cols['close']].shift(period))
                / result_df[price_cols['close']].shift(period) * 100
            ).round(4)

    # 3. 거래량 지표 (Volume Indicators)
    # OBV
    if 'obv' in selected_indicators:
        obv = (np.sign(result_df[price_cols['close']].diff()) * result_df[price_cols['volume']]).fillna(0).cumsum()
        result_df[f'{prefix}OBV'] = obv.round(4)

    # VR
    if 'vr' in selected_indicators:
        period = 20
        up_vol = result_df[price_cols['volume']].where(result_df[price_cols['close']].diff() > 0, 0)
        down_vol = result_df[price_cols['volume']].where(result_df[price_cols['close']].diff() < 0, 0)
        
        up_sum = up_vol.rolling(window=period).sum()
        down_sum = down_vol.rolling(window=period).sum()
        
        result_df[f'{prefix}VR'] = ((up_sum / (down_sum + epsilon)) * 100).round(4)

    # PVT
    if 'pvt' in selected_indicators:
        result_df[f'{prefix}PVT'] = (
            (result_df[price_cols['close']].diff() / result_df[price_cols['close']].shift(1))
            * result_df[price_cols['volume']]
        ).cumsum().round(4)

    # 4. 변동성 지표 (Volatility Indicators)
    # 볼린저 밴드
    if 'bollinger' in selected_indicators:
        period = 20
        middle = result_df[price_cols['close']].rolling(window=period).mean()
        std = result_df[price_cols['close']].rolling(window=period).std()
        
        result_df[f'{prefix}BB_{period}_upper'] = (middle + (2 * std)).round(4)
        result_df[f'{prefix}BB_{period}_middle'] = middle.round(4)
        result_df[f'{prefix}BB_{period}_lower'] = (middle - (2 * std)).round(4)
        result_df[f'{prefix}BB_{period}_bandwidth'] = (((result_df[f'{prefix}BB_{period}_upper'] - result_df[f'{prefix}BB_{period}_lower']) / middle) * 100).round(4)

    # ATR
    if 'atr' in selected_indicators:
        tr = pd.DataFrame()
        tr['hl'] = result_df[price_cols['high']] - result_df[price_cols['low']]
        tr['hc'] = abs(result_df[price_cols['high']] - result_df[price_cols['close']].shift(1))
        tr['lc'] = abs(result_df[price_cols['low']] - result_df[price_cols['close']].shift(1))
        tr['tr'] = tr[['hl', 'hc', 'lc']].max(axis=1)
        result_df[f'{prefix}ATR'] = tr['tr'].rolling(window=14).mean().round(4)

    # 5. 심리 지표 (Psychological Indicators)
    # 투자심리선 (PIS)
    if 'pis' in selected_indicators:
        period = 20
        up_days = (result_df[price_cols['close']].diff() > 0).rolling(window=period).sum()
        result_df[f'{prefix}PIS'] = ((up_days / period) * 100).round(4)

    # ADR
    if 'adr' in selected_indicators:
        period = 20
        advance = (result_df[price_cols['close']] > result_df[price_cols['close']].shift(1)).astype(int)
        decline = (result_df[price_cols['close']] < result_df[price_cols['close']].shift(1)).astype(int)
        
        adv_sum = advance.rolling(window=period).sum()
        dec_sum = decline.rolling(window=period).sum()
        
        result_df[f'{prefix}ADR'] = ((adv_sum / (dec_sum + epsilon)) * 100).round(4)

    # Handle infinite values and NaN
    result_df = result_df.replace([np.inf, -np.inf], 0).fillna(0)

    return result_df 