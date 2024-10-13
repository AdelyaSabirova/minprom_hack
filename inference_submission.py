import os
import json
import argparse
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument('-m', '--model', type=str, required=True, help='Путь к весам модели YOLO (.pt файл)')
    parser.add_argument('-src', '--source', type=str, required=True, help='Путь к папке с изображениями')
    parser.add_argument('-dst', '--destination', type=str, required=True, help='Путь к файлу для сохранения результатов (JSON)')
    parser.add_argument('--draw', action='store_true', help="Seve images with predictions")
    return parser.parse_args()

def load_model(weights_path):
    try:
        model = YOLO(weights_path, verbose= False)
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        exit(1)

def get_image_files(source_dir):
    supported_extensions = ('.jpg', '.png')
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(supported_extensions)]
    return files

def perform_inference(model, image_path, draw=False):
    results = model(image_path, conf=0.25, device='cuda:1', verbose=False)
    dets = results[0]
    objects = []
    if dets is not None:
        for i in range(len(dets.boxes.xyxyn)):
            obj = {
                "obj_class": str(int(dets.boxes.cls[i])),
                "x": str(dets.boxes.xyxyn[i][0].item()),
                "y": str(dets.boxes.xyxyn[i][1].item()),
                "width": str(dets.boxes.xywhn[i][2].item()),
                "height": str(dets.boxes.xywhn[i][3].item())
            }
            objects.append(obj)
    if draw:
        plotted = dets.plot()
        return objects, plotted
    return objects

def main():
    args = parse_arguments()

    model = load_model(args.model)
    image_files = get_image_files(args.source)
    
    if args.draw:
        p = Path(args.destination)
        (p.parent / p.stem).mkdir(exist_ok=True, parents=True)

    submission = []

    for img_file in tqdm(image_files, desc="Обработка изображений"):
        img_path = os.path.join(args.source, img_file)
        if args.draw:
            objects, plotted = perform_inference(model, img_path, args.draw)
            cv2.imwrite(str(p.parent / p.stem / img_file), plotted)
        objects = perform_inference(model, img_path)
        submission.append({
            "filename": img_file,
            "objects": objects if objects else []
        })

    try:
        with open(args.destination, 'w', encoding='utf-8') as f:
            json.dump(submission, f, ensure_ascii=False, indent=2)
        print(f"Результаты сохранены в {args.destination}")
    except Exception as e:
        print(f"Ошибка при сохранении файла JSON: {e}")

if __name__ == "__main__":
    main()
