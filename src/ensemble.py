import os
import glob
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion

def run_wbf_ensemble(prediction_dirs, weights=None, iou_thr=0.55, skip_box_thr=0.1, save_dir="./ensemble_output"):
    """
        WBF(Weighted Boxes Fusion) 알고리즘으로 정밀하게 융합합니다.
    
    Args:
        prediction_dirs (list): 'predict/labels' 폴더 경로 리스트 (예: [모델A_경로, 모델B_경로])
        weights (list): 각 모델별 가중치 (성능이 더 좋은 모델에 더 높은 투표권 부여)
        iou_thr (float): 상자들을 하나로 합칠 기준이 되는 겹침 비율(IoU) 역치값
        skip_box_thr (float): 지나치게 확신도가 낮은 바운딩 박스를 사전에 배제할 역치값
        save_dir (str): 앙상블 완료 후 최종 결합된 .txt 파일들이 저장될 경로
    """
    if weights is None:
        weights = [1] * len(prediction_dirs)
        
    # 1. 앙상블을 진행할 모든 고유 파일명(.txt) 확보
    all_txt_files = set()
    for p_dir in prediction_dirs:
        for txt_path in Path(p_dir).glob("*.txt"):
            all_txt_files.add(txt_path.name)
            
    txt_files = sorted(list(all_txt_files))
    print(f"🌟 [앙상블 엔진 가동]: 총 {len(prediction_dirs)}개 모델 결과물 융합 시작 ({len(txt_files)}개 파일 발견)")
    
    # 결과 저장 폴더 생성
    out_path = Path(save_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 2. 파일 단위로 순회하며 WBF 박스 결합 진행
    for txt_file in txt_files:
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for p_dir in prediction_dirs:
            file_path = Path(p_dir) / txt_file
            boxes = []
            scores = []
            labels = []
            
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # [교정] 텍스트 파싱 에러(공백, 깨진 데이터)로 인해 스크립트가 멈추는 것을 방지
                            try:
                                cls = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])
                                
                                # ... 좌표 변환 ... (x1, y1, x2, y2 계산부)
                                x1 = max(0.0, x - w/2)
                                y1 = max(0.0, y - h/2)
                                x2 = min(1.0, x + w/2)
                                y2 = min(1.0, y + h/2)
                                
                                # [교정] 신뢰도 누락 시 과도한 우대를 막기 위해 기본값을 1.0에서 0.5로 하향 조정
                                score = float(parts[5]) if len(parts) == 6 else 0.5
                                
                                boxes.append([x1, y1, x2, y2])
                                scores.append(score)
                                labels.append(cls)
                            except (ValueError, IndexError):
                                continue # 잘못된 형식의 라인은 건너뛰고 다음 라인 진행
                            
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
            
        # 3. 박스가 하나라도 존재하는 파일만 WBF 융합 진행
        if any(boxes_list):
            merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list, 
                weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
            )
            
            # 4. 결합된 결과를 다시 YOLO 포맷으로 역변환하여 저장
            out_file_path = out_path / txt_file
            with open(out_file_path, 'w', encoding='utf-8') as out_f:
                for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    x = x1 + w/2
                    y = y1 + h/2
                    
                    out_f.write(f"{int(label)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {score:.4f}\n")

    print(f"✅ [DONE] 앙상블 최종 완료! 저장 경로: {out_path.resolve()}")

if __name__ == "__main__":
    # 가중치 파일로 추론하여 공유해 준 결과 폴더 경로를 여기에 세팅
    # 예시 경로 구조:
    TEAM_PREDICTIONS = [
        "./predictions/member1_yolo11m_1280/labels",
        "./predictions/member2_yolo11x_640/labels",
        "./predictions/member3_yolo11m_augmented/labels"
    ]
    
    # 각 모델별 투표권 비율 (성능이 더 좋은 단일 모델에 2 또는 3을 부여하면 유리함)
    MODEL_WEIGHTS = [2, 1, 1] 
    
    # 앙상블 실행구
    # 준비되면 아래 주석을 풀고 실행
    # run_wbf_ensemble(prediction_dirs=TEAM_PREDICTIONS, weights=MODEL_WEIGHTS)