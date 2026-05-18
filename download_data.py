# 프로젝트 최상위 폴더/download.py
import kagglehub
import shutil
import os

def main():
    # 프로젝트 최상위 폴더 기준의 data 폴더 경로
    target_dir = os.path.join(os.getcwd(), 'data')
    
    # 이미 data 폴더가 있고 비어있지 않다면 실행 종료
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print("이미 data/ 폴더에 데이터셋이 존재하므로 다운로드를 건너뜁니다.")
        return
    
    print("Kaggle에서 데이터를 다운로드합니다.")
    cache_path = kagglehub.competition_download('ai11-level1-project')
    
    # 안전하게 안전장치 추가: data 폴더가 없으면 생성
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # 캐시 폴더 안의 내용물만 data/ 폴더 안으로 복사
    print("데이터를 data/ 폴더로 이동 중...")
    for item in os.listdir(cache_path):
        s = os.path.join(cache_path, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
            
    print(f"데이터 다운로드 및 이동 완료: {target_dir}")

if __name__ == "__main__":
    main()