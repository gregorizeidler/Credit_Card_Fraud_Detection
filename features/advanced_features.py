"""
Module for creating advanced features for fraud detection.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

# Add root directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_transaction_velocity_features(df, time_windows=[1, 6, 24], key_columns=['Card Number']):
    """
    Creates features related to transaction velocity.
    
    Args:
        df (pandas.DataFrame): DataFrame with transaction data.
        time_windows (list): List of time windows in hours to analyze.
        key_columns (list): Columns to identify unique entities (e.g., card, customer).
        
    Returns:
        pandas.DataFrame: DataFrame with the new features.
    """
    logger.info("Creating transaction velocity features")
    df_new = df.copy()
    
    # Make sure we have a date/time column
    if 'Transaction DateTime' not in df_new.columns:
        if all(col in df_new.columns for col in ['Transaction Date', 'Transaction Time']):
            df_new['Transaction DateTime'] = pd.to_datetime(
                df_new['Transaction Date'] + ' ' + df_new['Transaction Time'], 
                errors='coerce'
            )
        else:
            logger.error("Could not create Transaction DateTime, required columns not found")
            return df_new
    
    # Sort the DataFrame by date/time
    df_new = df_new.sort_values('Transaction DateTime')
    
    # Crie features de velocidade para cada janela de tempo
    for window in time_windows:
        # Nome da nova coluna
        col_name = f"tx_count_{window}h"
        
        # Para cada entidade (ex: cartão), conte transações na janela de tempo
        counts = []
        
        for _, group in df_new.groupby(key_columns):
            group = group.sort_values('Transaction DateTime')
            group_counts = []
            
            for idx, row in group.iterrows():
                current_time = row['Transaction DateTime']
                window_start = current_time - timedelta(hours=window)
                
                # Conte transações na janela anterior
                window_count = len(group[
                    (group['Transaction DateTime'] >= window_start) & 
                    (group['Transaction DateTime'] < current_time)
                ])
                
                group_counts.append(window_count)
            
            counts.extend(group_counts)
        
        # Adicione a nova coluna ao DataFrame
        df_new[col_name] = counts
    
    # Crie features derivadas
    for window in time_windows:
        count_col = f"tx_count_{window}h"
        # Flag para velocidades anormais
        df_new[f"high_velocity_{window}h"] = (
            df_new[count_col] > df_new[count_col].quantile(0.95)
        ).astype(int)
    
    logger.info(f"Features de velocidade criadas: {', '.join([f'tx_count_{w}h' for w in time_windows])}")
    return df_new

def create_behavioral_pattern_features(df):
    """
    Cria features que capturam padrões comportamentais para ajudar na detecção de fraudes.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados de transações.
        
    Returns:
        pandas.DataFrame: DataFrame com as novas features comportamentais.
    """
    logger.info("Criando features de padrões comportamentais")
    df_new = df.copy()
    
    # Garantir que temos as colunas necessárias
    required_cols = ['Transaction DateTime', 'Card Number', 'Amount', 'Merchant Category Code (MCC)']
    if not all(col in df_new.columns for col in required_cols):
        logger.warning("Algumas colunas necessárias para features comportamentais não foram encontradas")
        return df_new
    
    # Para cada cartão, calcule estatísticas de comportamento
    card_groups = df_new.groupby('Card Number')
    
    # Desvio do padrão de gasto (desvio z-score do valor)
    amount_means = card_groups['Amount'].transform('mean')
    amount_stds = card_groups['Amount'].transform('std')
    df_new['amount_zscore'] = np.where(
        amount_stds > 0,
        (df_new['Amount'] - amount_means) / amount_stds,
        0
    )
    
    # Marque valores extremos (outliers)
    df_new['unusual_amount'] = (abs(df_new['amount_zscore']) > 2.5).astype(int)
    
    # Frequência de uso por categoria de comerciante
    mcc_counts = df_new.groupby(['Card Number', 'Merchant Category Code (MCC)']).size().reset_index(name='mcc_count')
    mcc_counts = mcc_counts.rename(columns={'Merchant Category Code (MCC)': 'MCC'})
    
    # Normalizar contagens por cartão para obter frequências relativas
    card_total_txs = mcc_counts.groupby('Card Number')['mcc_count'].sum().reset_index(name='total_txs')
    mcc_counts = mcc_counts.merge(card_total_txs, on='Card Number')
    mcc_counts['mcc_frequency'] = mcc_counts['mcc_count'] / mcc_counts['total_txs']
    
    # Identificar categorias incomuns para cada cartão (baixa frequência)
    mcc_counts['unusual_mcc'] = (mcc_counts['mcc_frequency'] < 0.1).astype(int)
    
    # Mesclar de volta ao DataFrame original
    df_temp = df_new.merge(
        mcc_counts[['Card Number', 'MCC', 'mcc_frequency', 'unusual_mcc']],
        left_on=['Card Number', 'Merchant Category Code (MCC)'],
        right_on=['Card Number', 'MCC'],
        how='left'
    )
    
    # Preencher valores NaN
    df_new['mcc_frequency'] = df_temp['mcc_frequency'].fillna(0)
    df_new['unusual_mcc'] = df_temp['unusual_mcc'].fillna(1)  # Assume desconhecido como incomum
    
    # Criar score de risco baseado em comportamento
    df_new['behavior_risk_score'] = (
        df_new['unusual_amount'] * 0.6 +
        df_new['unusual_mcc'] * 0.4
    )
    
    logger.info("Features comportamentais criadas com sucesso")
    return df_new

def create_temporal_pattern_features(df):
    """
    Cria features que capturam padrões temporais para detecção de fraude.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados de transações.
        
    Returns:
        pandas.DataFrame: DataFrame com as novas features de padrões temporais.
    """
    logger.info("Criando features de padrões temporais")
    df_new = df.copy()
    
    # Garantir que temos a coluna de data/hora
    if 'Transaction DateTime' not in df_new.columns:
        if all(col in df_new.columns for col in ['Transaction Date', 'Transaction Time']):
            df_new['Transaction DateTime'] = pd.to_datetime(
                df_new['Transaction Date'] + ' ' + df_new['Transaction Time'], 
                errors='coerce'
            )
        else:
            logger.warning("Não foi possível criar coluna Transaction DateTime")
            return df_new
    
    # Extrair informações temporais
    df_new['hour'] = df_new['Transaction DateTime'].dt.hour
    df_new['day'] = df_new['Transaction DateTime'].dt.day
    df_new['day_of_week'] = df_new['Transaction DateTime'].dt.dayofweek
    df_new['is_weekend'] = (df_new['day_of_week'] >= 5).astype(int)
    
    # Definir períodos do dia
    df_new['period_of_day'] = pd.cut(
        df_new['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['madrugada', 'manha', 'tarde', 'noite'],
        include_lowest=True
    )
    
    # Para cada cartão, determinar o padrão típico de uso
    if 'Card Number' in df_new.columns:
        card_groups = df_new.groupby('Card Number')
        
        # Calcular distribuição típica de horas
        hour_counts = df_new.groupby(['Card Number', 'hour']).size().reset_index(name='hour_count')
        card_total = hour_counts.groupby('Card Number')['hour_count'].sum().reset_index(name='total_count')
        hour_counts = hour_counts.merge(card_total, on='Card Number')
        hour_counts['hour_frequency'] = hour_counts['hour_count'] / hour_counts['total_count']
        
        # Criar mapa de frequências para cada combinação de cartão e hora
        hour_freq_map = dict(zip(zip(hour_counts['Card Number'], hour_counts['hour']), hour_counts['hour_frequency']))
        
        # Aplicar o mapa ao DataFrame original
        df_new['hour_frequency'] = df_new.apply(
            lambda row: hour_freq_map.get((row['Card Number'], row['hour']), 0), 
            axis=1
        )
        
        # Identificar transações em horários incomuns para o cartão
        df_new['unusual_hour'] = (df_new['hour_frequency'] < 0.05).astype(int)
        
        # Repetir para dia da semana
        dow_counts = df_new.groupby(['Card Number', 'day_of_week']).size().reset_index(name='dow_count')
        dow_counts = dow_counts.merge(card_total, on='Card Number')
        dow_counts['dow_frequency'] = dow_counts['dow_count'] / dow_counts['total_count']
        
        # Criar mapa para dia da semana
        dow_freq_map = dict(zip(zip(dow_counts['Card Number'], dow_counts['day_of_week']), dow_counts['dow_frequency']))
        
        # Aplicar ao DataFrame
        df_new['dow_frequency'] = df_new.apply(
            lambda row: dow_freq_map.get((row['Card Number'], row['day_of_week']), 0), 
            axis=1
        )
        
        # Identificar transações em dias incomuns
        df_new['unusual_day'] = (df_new['dow_frequency'] < 0.05).astype(int)
    
    # Feature combinada de risco temporal
    if all(col in df_new.columns for col in ['unusual_hour', 'unusual_day']):
        df_new['temporal_risk_score'] = (
            df_new['unusual_hour'] * 0.7 +
            df_new['unusual_day'] * 0.3
        )
    
    # Marcar transações de alto risco (madrugada/noite em finais de semana)
    df_new['high_risk_time_pattern'] = (
        ((df_new['period_of_day'] == 'madrugada') | (df_new['period_of_day'] == 'noite')) &
        (df_new['is_weekend'] == 1)
    ).astype(int)
    
    logger.info("Features de padrões temporais criadas com sucesso")
    return df_new

def create_fraud_detection_score(df):
    """
    Cria um score composto de risco de fraude combinando múltiplas features.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados de transações e features derivadas.
        
    Returns:
        pandas.DataFrame: DataFrame com o score de risco de fraude adicionado.
    """
    logger.info("Criando score de risco de fraude")
    df_new = df.copy()
    
    # Lista de features potenciais (use as que estiverem disponíveis)
    risk_factors = {
        'unusual_amount': 0.25,
        'unusual_mcc': 0.15,
        'unusual_hour': 0.20,
        'unusual_day': 0.10,
        'high_risk_time_pattern': 0.15,
        'high_velocity_24h': 0.15
    }
    
    # Verificar quais features estão disponíveis
    available_factors = [f for f in risk_factors.keys() if f in df_new.columns]
    
    if not available_factors:
        logger.warning("Nenhuma feature de risco disponível para criar score de fraude")
        return df_new
    
    # Normalizar pesos para as features disponíveis
    total_weight = sum(risk_factors[f] for f in available_factors)
    normalized_weights = {f: risk_factors[f] / total_weight for f in available_factors}
    
    # Criar score ponderado
    df_new['fraud_risk_score'] = 0
    for factor, weight in normalized_weights.items():
        df_new['fraud_risk_score'] += df_new[factor] * weight
    
    # Escalar para 0-100 para facilitar interpretação
    df_new['fraud_risk_score'] = df_new['fraud_risk_score'] * 100
    
    # Categorizar score de risco
    df_new['risk_category'] = pd.cut(
        df_new['fraud_risk_score'],
        bins=[0, 25, 50, 75, 100],
        labels=['baixo', 'medio', 'alto', 'muito_alto'],
        include_lowest=True
    )
    
    logger.info("Score de risco de fraude criado com sucesso")
    return df_new

def create_all_advanced_features(df):
    """
    Aplica todas as funções de criação de features avançadas.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados de transações.
        
    Returns:
        pandas.DataFrame: DataFrame com todas as features avançadas adicionadas.
    """
    logger.info("Iniciando criação de todas as features avançadas")
    
    # Aplicar funções de features em sequência
    df_new = df.copy()
    df_new = create_transaction_velocity_features(df_new)
    df_new = create_behavioral_pattern_features(df_new)
    df_new = create_temporal_pattern_features(df_new)
    df_new = create_fraud_detection_score(df_new)
    
    logger.info(f"Criação de features avançadas concluída. Total de colunas: {len(df_new.columns)}")
    return df_new 
