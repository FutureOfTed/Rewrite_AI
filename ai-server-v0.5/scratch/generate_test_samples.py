import json
import os
import random

def generate_wave(scenario="normal", duration=100):
    frames = []
    
    # 기본값 설정
    base_acc = 0.7
    base_hit_prob = 0.1
    is_dead_at_end = False
    fail_safe = False
    
    if scenario == "pro":
        base_acc = 0.9
        base_hit_prob = 0.02
    elif scenario == "newbie":
        base_acc = 0.3
        base_hit_prob = 0.3
    elif scenario == "panic_churn":
        base_acc = 0.6
        fail_safe = True
        is_dead_at_end = True
    elif scenario == "chain_hit_churn":
        base_acc = 0.5
        fail_safe = True
        is_dead_at_end = True

    current_hp = 150
    max_hp = 150

    for sec in range(1, duration + 1):
        # 시나리오별 동적 변화
        current_sec_acc = base_acc
        current_sec_hit_taken = 0
        current_sec_apm = 200 + random.randint(-20, 20)
        
        # 패닉 난사 시나리오: 마지막 10초간 명중률 폭락 및 APM 폭증
        if scenario == "panic_churn" and sec > (duration - 10):
            current_sec_acc = base_acc * 0.3
            current_sec_apm = 450 + random.randint(0, 30)
            
        # 연쇄 피격 시나리오: 마지막 5초간 피격 집중
        if scenario == "chain_hit_churn" and sec > (duration - 5):
            current_sec_hit_taken = 20 # 초당 20씩 깎임 (5초면 100)
        else:
            if random.random() < base_hit_prob:
                current_sec_hit_taken = 5

        hp_lost = current_sec_hit_taken
        current_hp = max(0, current_hp - hp_lost)
        
        # frames 데이터 조립
        frames.append({
            "sec": sec,
            "atk_clicks_total": 10,
            "atk_clicks_hit": int(10 * current_sec_acc),
            "enemy_atk_spawned": 10,
            "hitbox_collisions": 1 if current_sec_hit_taken > 0 else 0,
            "base_dmg_expected": 30.0,
            "actual_dmg_dealt": 120.0 if current_sec_acc > 0.5 else 40.0,
            "hp_lost": hp_lost,
            "max_hp": max_hp,
            "apm": current_sec_apm
        })
        
        if current_hp <= 0:
            break

    return {
        "log_id": f"test_log_{scenario}_{random.randint(100, 999)}",
        "wave_meta": {
            "clear_time_sec": float(len(frames)),
            "fail_safe": fail_safe,
            "floor": 1,
            "wave": 1
        },
        "time_series_frames": frames
    }

def main():
    scenarios = ["pro", "newbie", "panic_churn", "chain_hit_churn"]
    all_waves = []
    
    # 각 시나리오별로 5개씩 생성 (총 20개 웨이브)
    for s in scenarios:
        for _ in range(5):
            all_waves.append(generate_wave(s))
            
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/test_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_waves, f, indent=2, ensure_ascii=False)
    
    print(f"정교한 테스트 데이터 생성 완료: scratch/test_dataset.json (총 {len(all_waves)}개 웨이브)")

if __name__ == "__main__":
    main()
