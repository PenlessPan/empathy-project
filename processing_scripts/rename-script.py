import os
import shutil
from pathlib import Path


def copy_and_rename_files(source_dir, target_dir):
    # Hebrew to English mapping
    camera_angles = {
        'עליון': 'top',
        'פנים': 'face',
        'תקרה': 'ceiling',
        'תיקרה': 'ceiling'
    }

    # Create target directory
    os.makedirs(target_dir, exist_ok=True)

    # Process all files in the movies directory
    for session_num in range(101, 141):
        if session_num in [117, 118, 139]:
            continue

        session_dir = os.path.join(source_dir, str(session_num))
        if not os.path.exists(session_dir):
            continue

        # Create session directory in target
        new_session_dir = os.path.join(target_dir, str(session_num))
        os.makedirs(new_session_dir, exist_ok=True)

        # Process each file in the session directory
        for filename in os.listdir(session_dir):
            if not filename.endswith('.mp4'):
                continue

            # Special case for file without sound
            if 'ללא קול' in filename:
                new_name = f"{session_num}_face_no_sound.mp4"
            else:
                # Regular file processing
                for heb, eng in camera_angles.items():
                    if heb in filename:
                        new_name = f"{session_num}_{eng}.mp4"
                        break
                else:
                    continue  # Skip if no matching angle found

            old_path = os.path.join(session_dir, filename)
            new_path = os.path.join(new_session_dir, new_name)

            # Copy with new name
            print(f"Copying: {filename} -> {new_name}")
            shutil.copy2(old_path, new_path)

    return "File copying completed. New files available in target directory."


# Usage
if __name__ == "__main__":
    source_dir = "movies"  # Replace with actual source path
    target_dir = "processed_movies"  # Replace with desired target path
    result = copy_and_rename_files(source_dir, target_dir)
    print(result)