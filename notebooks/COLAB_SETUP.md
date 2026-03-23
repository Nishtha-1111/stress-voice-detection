# Google Colab Setup Instructions

## Step 1: Upload to Google Drive
1. Open [Google Drive](https://drive.google.com)
2. Create a folder called `stress_voice_detection`
3. Upload your entire project folder to this location

## Step 2: Open in Colab
1. Go to [Google Colab](https://colab.research.google.com)
2. Select "File" → "Upload notebook"
3. Upload the `ravdess_training_pipeline.ipynb` file

## Step 3: Enable GPU
1. In Colab, go to "Runtime" → "Change runtime type"
2. Select "GPU" as Hardware accelerator
3. Click "Save"

## Step 4: Run Training
Run each cell in the notebook sequentially.

## Step 5: Download Models
After training completes:
1. The models will be saved in your Google Drive
2. Download them to your local `models/` folder