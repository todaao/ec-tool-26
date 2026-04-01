import pandas as pd
import json
import sys
import numpy as np
from pathlib import Path

BRAND_MAP = {
    'セット：MF': '2.MilleFée', 'セット：TOKOTOYZ': '1.TOKOTOYZ',
    'セット：JL': 'jill leen.', 'セット：CD': 'jill leen.',
    'セット：FCC': '4.fractionalCC', 'セット：HiCA': 'HiCA', 'セット：EE': 'Emery Emily',
}
SET_BRANDS = set(BRAND_MAP.keys())

def load_and_normalize(xlsx_path):
    df = pd.read_excel(xlsx_path)
    df['日付'] = pd.to_datetime(df['日付'])
    df['ブランド正規化'] = df['ブランド名'].map(lambda x: BRAND_MAP.get(x, x))
    df['is_set_parent'] = df['ブランド名'].isin(SET_BRANDS)
    df['is_set_detail'] = (df['売上高'] == 0) & (~df['ブランド名'].isin(SET_BRANDS))
    return df

def load_targets(target_xlsx):
    raw = pd.read_excel(target_xlsx)
    mall_rows = {
        '★自社国内EC':0,'1.Amazon':3,'2.Qoo10':6,'3.楽天':9,
        '4.TikTokShop':12,'ZOZO':15,'yahoo':18,'LINE GIFT':21,'TANP':24,'メルカリ':27,
    }
    targets = {}
    for mall, row_idx in mall_rows.items():
        targets[mall] = {}
        for m in range(1, 13):
            try:
                val = raw.iloc[row_idx][m]
                targets[mall][str(m)] = float(val) if pd.notna(val) else 0
            except: targets[mall][str(m)] = 0
    return targets

# ── 配列形式でJSONを圧縮する関数
def to_compact(df, cols, round_cols=None, int_cols=None):
    """DataFrameをヘッダー+行配列形式で返す（キー名の重複排除）"""
    round_cols = round_cols or {}
    int_cols = int_cols or []
    rows = []
    for _, r in df.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if c in int_cols:
                row.append(int(v))
            elif c in round_cols:
                row.append(round(float(v), round_cols[c]))
            else:
                row.append(v if not isinstance(v, float) or not np.isnan(v) else 0)
        rows.append(row)
    return {'cols': cols, 'rows': rows}

def aggregate(df):
    df = df.copy()
    df['date_str'] = df['日付'].dt.strftime('%Y-%m-%d')
    df['year']     = df['日付'].dt.year
    df['month']    = df['日付'].dt.month
    df['week_end'] = df['日付'].dt.to_period('W-SAT').apply(lambda p: p.end_time.strftime('%Y-%m-%d'))

    df_amount = df[df['売上高'] > 0].copy()
    df_qty    = df[(df['売上高'] > 0) | df['is_set_detail']].copy()

    # 1. 日次 × モール
    daily = {}
    for (d, m), v in df_amount.groupby(['date_str','モール'])['売上高'].sum().items():
        daily.setdefault(d, {'a':{},'q':{}})
        daily[d]['a'][m] = round(float(v)/10000, 4)
    for (d, m), v in df_qty.groupby(['date_str','モール'])['売上個数'].sum().items():
        daily.setdefault(d, {'a':{},'q':{}})
        daily[d]['q'][m] = int(v)

    # 2. 日次 × モール × ブランド
    dba, dbq = {}, {}
    for (d,m,b), v in df_amount.groupby(['date_str','モール','ブランド正規化'])['売上高'].sum().items():
        dba.setdefault(d,{}).setdefault(m,{})[b] = round(float(v)/10000,4)
    for (d,m,b), v in df_qty.groupby(['date_str','モール','ブランド正規化'])['売上個数'].sum().items():
        dbq.setdefault(d,{}).setdefault(m,{})[b] = int(v)

    # 3. 週次 × モール
    weekly = {}
    for (w, m), v in df_amount.groupby(['week_end','モール'])['売上高'].sum().items():
        weekly.setdefault(w, {'a':{},'q':{}})
        weekly[w]['a'][m] = round(float(v)/10000, 4)
    for (w, m), v in df_qty.groupby(['week_end','モール'])['売上個数'].sum().items():
        weekly.setdefault(w, {'a':{},'q':{}})
        weekly[w]['q'][m] = int(v)

    # 4. ピボット（週次止まり・配列形式で圧縮）
    bp_keys = ['month','week_end','ブランド正規化','商品名','型番','モール']
    mp_keys = ['month','week_end','モール','ブランド正規化','商品名','型番']

    bpa = df_amount.groupby(bp_keys)['売上高'].sum().reset_index()
    bpa['am'] = (bpa['売上高']/10000).round(4)
    bpq = df_qty.groupby(bp_keys)['売上個数'].sum().reset_index()
    bp = pd.merge(bpa.drop(columns=['売上高']), bpq, on=bp_keys, how='outer').fillna(0)
    bp['month'] = bp['month'].astype(int)
    bp['売上個数'] = bp['売上個数'].astype(int)
    bp_cols = ['month','week_end','ブランド正規化','商品名','型番','モール','am','売上個数']
    brand_pivot = to_compact(bp, bp_cols, round_cols={'am':4}, int_cols=['month','売上個数'])

    mpa = df_amount.groupby(mp_keys)['売上高'].sum().reset_index()
    mpa['am'] = (mpa['売上高']/10000).round(4)
    mpq = df_qty.groupby(mp_keys)['売上個数'].sum().reset_index()
    mp = pd.merge(mpa.drop(columns=['売上高']), mpq, on=mp_keys, how='outer').fillna(0)
    mp['month'] = mp['month'].astype(int)
    mp['売上個数'] = mp['売上個数'].astype(int)
    mp_cols = ['month','week_end','モール','ブランド正規化','商品名','型番','am','売上個数']
    mall_pivot = to_compact(mp, mp_cols, round_cols={'am':4}, int_cols=['month','売上個数'])

    # 5. 週次・月次 商品別（配列形式）
    wpa = df_amount.groupby(['week_end','モール','ブランド正規化','商品名','型番'])['売上高'].sum().reset_index()
    wpa['am'] = (wpa['売上高']/10000).round(4)
    wpq = df_qty.groupby(['week_end','モール','ブランド正規化','商品名','型番'])['売上個数'].sum().reset_index()
    wp = pd.merge(wpa.drop(columns=['売上高']), wpq, on=['week_end','モール','ブランド正規化','商品名','型番'], how='outer').fillna(0)
    wp['売上個数'] = wp['売上個数'].astype(int)
    wp_cols = ['week_end','モール','ブランド正規化','商品名','型番','am','売上個数']
    weekly_product = to_compact(wp, wp_cols, round_cols={'am':4}, int_cols=['売上個数'])

    mpa2 = df_amount.groupby(['month','モール','ブランド正規化','商品名','型番'])['売上高'].sum().reset_index()
    mpa2['am'] = (mpa2['売上高']/10000).round(4)
    mpq2 = df_qty.groupby(['month','モール','ブランド正規化','商品名','型番'])['売上個数'].sum().reset_index()
    mp2 = pd.merge(mpa2.drop(columns=['売上高']), mpq2, on=['month','モール','ブランド正規化','商品名','型番'], how='outer').fillna(0)
    mp2['month'] = mp2['month'].astype(int)
    mp2['売上個数'] = mp2['売上個数'].astype(int)
    mp2_cols = ['month','モール','ブランド正規化','商品名','型番','am','売上個数']
    monthly_product = to_compact(mp2, mp2_cols, round_cols={'am':4}, int_cols=['month','売上個数'])

    # 6. 新商品週次（配列形式）
    npw = df_amount.groupby(['week_end','商品名','モール']).agg(am=('売上高', lambda x: round(x.sum()/10000,4)), qt=('売上個数','sum')).reset_index()
    npw['qt'] = npw['qt'].astype(int)
    new_product_weekly = to_compact(npw, ['week_end','商品名','モール','am','qt'], round_cols={'am':4}, int_cols=['qt'])

    # 7. 異常検知
    anomalies = detect_anomalies(df_amount)

    return {
        'daily': daily, 'dba': dba, 'dbq': dbq, 'weekly': weekly,
        'brand_pivot': brand_pivot, 'mall_pivot': mall_pivot,
        'weekly_product': weekly_product, 'monthly_product': monthly_product,
        'new_product_weekly': new_product_weekly, 'anomalies': anomalies,
    }

def detect_anomalies(df_amount):
    # SKU単位（商品名＋型番）で集計
    daily_prod = df_amount.groupby(['date_str','商品名','型番','ブランド正規化'])['売上高'].sum().reset_index()
    daily_prod['date'] = pd.to_datetime(daily_prod['date_str'])
    results = []
    for (prod, sku), grp in daily_prod.groupby(['商品名','型番']):
        grp = grp.sort_values('date').reset_index(drop=True)
        if len(grp) < 5: continue
        vals = grp['売上高'].values.astype(float)
        mean, std = np.mean(vals), np.std(vals)
        if std == 0: continue
        brand = grp['ブランド正規化'].iloc[0]
        for i, row in grp.iterrows():
            v, z, d = row['売上高'], (row['売上高']-mean)/std, row['date_str']
            if z >= 3.0:
                future = grp[grp['date'] > row['date']].head(3)
                if len(future)>=1 and (future['売上高']>=mean*1.5).all():
                    results.append({'type':'継続急増','icon':'📈','date':d,'product':prod,'sku':sku,'brand':brand,'amount_man':round(v/10000,2),'z':round(z,2),'hypothesis':'SNSバズ・メディア掲載・インフルエンサー投稿'})
                else:
                    results.append({'type':'一日スパイク','icon':'⚡','date':d,'product':prod,'sku':sku,'brand':brand,'amount_man':round(v/10000,2),'z':round(z,2),'hypothesis':'TOKOTOYZ予約確定タイミング・限定キャンペーン流入'})
            if z <= -2.0:
                prev5 = grp[grp['date'] < row['date']].tail(5)
                if len(prev5)>=3 and prev5['売上高'].mean()>=mean*1.3:
                    results.append({'type':'在庫切れ候補','icon':'📉','date':d,'product':prod,'sku':sku,'brand':brand,'amount_man':round(v/10000,2),'z':round(z,2),'hypothesis':'好調中の急減＝欠品の疑い'})
    daily_total = df_amount.groupby('date_str')['売上高'].sum().reset_index()
    if len(daily_total) >= 5:
        vals = daily_total['売上高'].values.astype(float)
        mean, std = np.mean(vals), np.std(vals)
        if std > 0:
            for _, row in daily_total.iterrows():
                z = (row['売上高']-mean)/std
                if abs(z) >= 2.5:
                    results.append({'type':'全体異常' if z>0 else '全体低迷','icon':'🌐','date':row['date_str'],'product':'（全商品合計）','brand':'-','amount_man':round(row['売上高']/10000,2),'z':round(z,2),'hypothesis':'全体的な売上変動（セール・障害等）'})
    results.sort(key=lambda x: x['date'], reverse=True)
    return results

if __name__ == '__main__':
    sales_xlsx  = sys.argv[1] if len(sys.argv)>1 else '2026実績_ツール処理用_.xlsx'
    target_xlsx = sys.argv[2] if len(sys.argv)>2 else '2026年売り上げ目標.xlsx'
    out         = sys.argv[3] if len(sys.argv)>3 else 'ec_data.json'

    print(f'Loading {sales_xlsx}...')
    df = load_and_normalize(sales_xlsx)
    print(f'Total rows: {len(df)}')
    print('Loading targets...')
    targets = load_targets(target_xlsx)
    print('Aggregating...')
    data = aggregate(df)
    data['targets'] = targets
    data['generated_at'] = pd.Timestamp.now().isoformat()
    print(f'Anomalies: {len(data["anomalies"])}')

    with open(out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',',':'))

    kb = Path(out).stat().st_size / 1024
    print(f'✅ {out} ({kb:.0f} KB)')
