from notebook.PM.test import load_yolo_data_config, validate_yolo_data_config, verify_yolo_conversion


print("\n[STEP 1] YOLO 데이터 구성 파일 로딩")

DATA_YAML = "/Users/apple/Desktop/project_team4/data/PROJECT_TEAM4/data.yaml"

data_config = load_yolo_data_config(DATA_YAML)
validate_yolo_data_config(data_config)
    
print(f" └─ YOLO data.yaml loaded: root={data_config['root']}")
print(f" └─ Paths: train_dir={data_config['train_dir']}, val_dir={data_config['val_dir']}")
print(f" └─ Classes: nc={data_config['nc']} ({data_config['nc']}개 알약 클래스 확인)")
print(" └─ YOLO 데이터 구성 확인이 완료되었습니다.")

verify_yolo_conversion(data_config)