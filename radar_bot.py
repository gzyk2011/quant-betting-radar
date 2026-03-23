import penaltyblog as pb
import pandas as pd
import requests
import os
import warnings

warnings.filterwarnings("ignore")

TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

def send_tg(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        print(f"DEBUG: {text}")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        print(f"TG 发送失败: {e}")

def run_radar():
    print("🚀 启动全球联赛全能扫描...")
    # 发送一个心跳信号
    send_tg("📡 *量化雷达已启动*：正在扫描全球活跃联赛（含美职联/五大联赛）...")
    
    # 联赛配置：包含欧洲五大联赛和美超(MLS)
    # 注意：MLS 通常按自然年计算赛季，2026年赛季用 "2026" 或 "2025-2026" 视数据源更新而定
    leagues = [
        ("USA MLS", "2026"),                # 正在赛季中！
        ("ENG Premier League", "2025-2026"), # 下周回归
        ("ENG Championship", "2025-2026"),
        ("GER 1. Bundesliga", "2025-2026"),
        ("ITA Serie A", "2025-2026"),
        ("ESP La Liga", "2025-2026"),
        ("FRA Ligue 1", "2025-2026")
    ]
    
    found_any = False
    report = ["🚨 *实时价值投注建议* 🚨\n"]

    for label, season in leagues:
        try:
            print(f"正在拉取: {label} ({season})...")
            df = pb.scrapers.FootballData(label, season).get_fixtures()
            
            # 区分历史(训练)与未来(预测)
            df_train = df.dropna(subset=['fthg', 'ftag'])
            df_upcoming = df[df['fthg'].isna()]
            
            if df_upcoming.empty:
                print(f"--- {label}: 暂无未来赛程 ---")
                continue

            # 训练模型
            model = pb.models.DixonColesGoalModel(
                df_train["goals_home"], df_train["goals_away"], 
                df_train["team_home"], df_train["team_away"]
            )
            model.fit(use_gradient=True, minimizer_options={"disp": False})

            # 扫描未来比赛
            for _, match in df_upcoming.iterrows():
                h, a = match['team_home'], match['team_away']
                # 获取 Bet365 赔率 (部分联赛可能在列名上有细微差别，用 get 保护)
                o_h = match.get('b365_h')
                
                if pd.notna(o_h) and 1.5 <= o_h <= 4.0:
                    pred = model.predict(h, a)
                    # 策略：Edge > 3% (稍微降低门槛确保你能看到信号)
                    edge = pred.home_win - (1/o_h)
                    
                    if edge > 0.03:
                        found_any = True
                        report.append(f"🏆 *{label}*\n⚽ {h} vs {a}\n📈 建议: 主胜 | 赔率: `{o_h}`\n🔥 优势: `{edge:.2%}`\n")
                        
        except Exception as e:
            print(f"跳过 {label}: 错误 {str(e)[:30]}")
            continue

    if found_any:
        send_tg("\n".join(report))
    else:
        send_tg("✅ *扫描完成*：当前全球活跃联赛暂无符合策略的盘口，建议继续观望。")

if __name__ == "__main__":
    run_radar()
