# Check if a folder path is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <folder_path> <folder_name>"
    exit 1
fi

FOLDER_PATH="$1"
FOLDER_NAME="$2"

# Check if the folder exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo "Error: Folder '$FOLDER_PATH' not found."
    exit 1
fi

# Iterate over every .png image in the folder
for png_file in "$FOLDER_PATH"/"$FOLDER_NAME"/*.png; do
    # Check if folder with same name exists
    filename=$(basename "$png_file" .png)

    echo "$png_file"
    # Check if there are any matching files
    if [ -e "$png_file" ]; then

        python run.py --img_path "$png_file" --half_precision --output_format .obj --folder "$FOLDER_NAME"

    fi

done
