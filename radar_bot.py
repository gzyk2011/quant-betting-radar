import penaltyblog as pb
import pandas as pd
import requests
import os
import warnings

warnings.filterwarnings("ignore")

# 环境变量获取
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

def send_tg(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        print(f"DEBUG Console: {text}")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        print(f"发送 TG 失败: {e}")

def run_radar():
    print("🚀 启动五大联赛深度扫描...")
    # 发送启动信号，确保你能在 TG 看到它动了
    send_tg("📡 *量化雷达已唤醒*，正在扫描全网盘口...")
    
    messages = []
    
    # 联赛清单
    leagues = [
        ("ENG Premier League", "2023-2024"),
        ("ENG Championship", "2023-2024"),
        ("GER Bundesliga", "2023-2024"),
        ("ITA Serie A", "2023-2024"),
        ("ESP La Liga", "2023-2024"),
        ("FRA Ligue 1", "2023-2024")
    ]
    
    for label, season in leagues:
        try:
            print(f"正在拉取: {label}...")
            df = pb.scrapers.FootballData(label, season).get_fixtures()
            
            # 数据清洗
            df_train = df.dropna(subset=['fthg', 'ftag'])
            df_upcoming = df[df['fthg'].isna()]
            
            if df_train.empty or df_upcoming.empty:
                print(f"跳过 {label}: 无可训练数据或未来赛程")
                continue

            # 训练模型
            model = pb.models.DixonColesGoalModel(
                df_train["goals_home"], df_train["goals_away"], 
                df_train["team_home"], df_train["team_away"]
            )
            model.fit(use_gradient=True, minimizer_options={"disp": False})

            for _, match in df_upcoming.iterrows():
                h, a = match['team_home'], match['team_away']
                o_h = match.get('b365_h')
                
                # 策略过滤：黄金区间 + 门槛下调至 2% (方便你测试收到消息)
                if pd.notna(o_h) and 1.4 <= o_h <= 4.0:
                    pred = model.predict(h, a)
                    edge = pred.home_win - (1/o_h)
                    
                    if edge > 0.02: 
                        messages.append(f"⚽ *{label}*\n{h} vs {a}\n建议: 主胜 | 赔率: `{o_h}`\n优势: `{edge:.2%}`\n")
                        
        except Exception as e:
            print(f"分析 {label} 时出错: {str(e)[:50]}") # 只打印核心错误
            continue

    if messages:
        header = "🚨 *发现价值投注场次* 🚨\n"
        send_tg(header + "\n".join(messages))
    else:
        send_tg("✅ *扫描完成*：当前暂无符合策略的场次。")

if __name__ == "__main__":
    run_radar()
