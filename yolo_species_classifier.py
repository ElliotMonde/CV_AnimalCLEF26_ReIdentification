import os
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import shutil

TRAIN_FP = os.path.join(os.getcwd(), "data", "train")
VAL_FP = os.path.join(os.getcwd(), "data", "val")


######### data processing: segmentate images ###########

def generate_bbox_ground_truth(image, model=YOLO("yolo26n.pt")):
    results = model(image)
    r = results[0]  # first result only
    # take most confident box in results
    polygon = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0] # fallback polygon
    if len(r.boxes) > 0:
        box = r.boxes.xywhn[0].cpu().numpy()
        xc, yc, w, h = box

        xl = xc - w / 2
        xr = xc + w / 2
        yt = yc - h / 2
        yb = yc + h / 2

        polygon = [xl, yt, xr, yt, xr, yb, xl, yb]
    
    return polygon


## data processing: unprocessed --> /images and /labels
def sort_images_train_test():

    # make destination root/train and root/val directories
    Path(TRAIN_FP).mkdir(parents=True, exist_ok=True)
    Path(VAL_FP).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(os.path.join(os.getcwd(), "data", "unprocessed", "metadata.csv"))

    cols = ["image_id", "path", "dataset", "identity"]

    train_df :pd.DataFrame = df.loc[df["split"] == "train", cols]
    val_df :pd.DataFrame = df.loc[df["split"] == "test", cols]

    # process species tag
    def get_species(x):
        return x.lower().split("id")[0]

    train_df["dataset"] = train_df["dataset"].apply(get_species)
    val_df["dataset"] = val_df["dataset"].apply(get_species)

    # normalized bbox: top-left, top-right, bottom-right, bottom-left
    vertices = " " + " ".join(map(str, [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]))

    # add label for each sample
    labels_map = {"lynx": 0, "salamander": 1, "seaturtle": 2, "texashornedlizards": 3}

    # load bbox model
    bbox_model = YOLO("yolo26n.pt")

    def save_labels(df: pd.DataFrame, dest: Path):
    # generate gt bbox
        for _,row in df.iterrows():
            src_path = Path(os.getcwd()) / "data" / "unprocessed" / row["path"]

            polygon = generate_bbox_ground_truth(str(src_path), bbox_model)

            # create label content
            class_id = labels_map[str(row["dataset"])]

            # get label text
            label_text = f"{class_id} {' '.join(map(str, polygon))}"

            img_dir = dest / "images"
            lbl_dir = dest / "labels"

            # save image in dest images folder
            shutil.copy(src_path, img_dir / f"{row['image_id']}.jpg")

            # save label txt in labels folder
            with open(lbl_dir / f"{row['image_id']}.txt", "w") as f:
                f.write(label_text)
    
    save_labels(train_df, Path(TRAIN_FP))
    save_labels(val_df, Path(VAL_FP))


####### train species classifier #########
from datetime import datetime


def train_yolo(model=YOLO("yolo11m-seg.pt"), data:str="data.yaml"):
    proj = os.path.join(os.getcwd(),"sutd_cv_project")
    name = f"exp_yolo_species_classification_{datetime.now().day}_{datetime.now().month}_{datetime.now().hour}"
    results = model.train(
        data=data,
        epochs=50,
        imgsz=640,
        batch=8,
        device='mps',
        name=name,
        project=proj,
        exist_ok=True,
        augment=True,
    )
    if results is not None:
        print(f"--- Training Completed ---\nproject: {proj}, \n name:{name}")
        print(f"Results saved to: {results.save_dir}")
        print(f"Model Fitness Score: {results.fitness:.4f}")
        metrics = model.val()
        
        # Print specific metrics
        print(f"mAP50-95: {metrics.box.map}")    # Mean Average Precision (IoU 0.50:0.95)
        print(f"mAP50:    {metrics.box.map50}")  # mAP at IoU 0.50
        print(f"mAP75:    {metrics.box.map75}")  # mAP at IoU 0.75
        print(f"Precision: {metrics.box.mp}")    # Mean Precision
        print(f"Recall:    {metrics.box.mr}")    # Mean Recall
        print(f"F1 Score per class: {metrics.box.f1}")
        print(f"Mean F1 Score: {metrics.box.f1.mean()}")
    return results
