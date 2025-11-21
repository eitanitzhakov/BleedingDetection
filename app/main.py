from pipeline.train_pipeline import TrainPipeline


def main():

    data_dir = r"C:\Users\eitan\Downloads\rsna\stage_2_train"
    csv_path = r"C:\Users\eitan\Downloads\rsna\stage_2_train.csv"

    pipeline = TrainPipeline(
        data_dir=data_dir,
        csv_path=csv_path,
        batch_size=2,
        val_split=0.1,
        checkpoint_path="checkpoints/best_auc.h5",
        final_model_path="saved_models/final_model.h5"
    )

    print("Testing one batch...")
    images, labels = pipeline.train_loader[0]
    preds = pipeline.model.model.predict(images)
    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Preds:", preds.shape)

    print("\nStage 1 training...")
    pipeline.train_stage1(epochs=5)

    print("\nFine tuning...")
    pipeline.fine_tune(num_layers=40, epochs=10)

    pipeline.save_final()

    print("\nTraining complete! Final model saved.")


if __name__ == "__main__":
    main()
