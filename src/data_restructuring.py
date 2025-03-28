import os
import shutil
import xml.etree.ElementTree as ET
import re


def delete_images_without_xml(base_image_folder, base_xml_folder):
    deleted_count = 0
    
    for root, _, files in os.walk(base_image_folder):
        # Berechnen des entsprechenden XML-Ordners
        relative_path = os.path.relpath(root, base_image_folder)
        xml_folder = os.path.join(base_xml_folder, relative_path)

        if not os.path.exists(xml_folder):
            continue  # Falls der entsprechende XML-Ordner nicht existiert, überspringen

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image_path = os.path.join(root, file)
                xml_file = os.path.splitext(file)[0] + '.xml'
                xml_path = os.path.join(xml_folder, xml_file)

                if not os.path.exists(xml_path):
                    os.remove(image_path)
                    deleted_count += 1
                    print(f"Deleted image: {image_path}")

    print(f"\nTotal images deleted: {deleted_count}")


def update_paths(base_annotations_folder, new_folder_name):
    for root, _, files in os.walk(base_annotations_folder):
        for file in files:
            if file.endswith(".xml"):
                xml_file_path = os.path.join(root, file)

                # XML-Datei öffnen und auslesen
                with open(xml_file_path, 'rb') as xml_file:
                    xml_content = xml_file.read()

                root_xml = ET.fromstring(xml_content)

                # Aktualisieren des <folder>-Tags
                folder_element = root_xml.find("folder")
                if folder_element is not None:
                    folder_element.text = new_folder_name

                # Aktualisieren des <filename>-Tags
                filename_element = root_xml.find("filename")
                if filename_element is not None:
                    new_filename = os.path.splitext(file)[0] + ".png"  # <- Hier korrekter Dateiname
                    filename_element.text = new_filename

                # Speichern der aktualisierten XML-Datei
                tree = ET.ElementTree(root_xml)
                tree.write(xml_file_path, encoding="utf-8", xml_declaration=False)
                print(f"Updated XML: {xml_file_path}")


def copy_all_files(src_folder, dest_folder):
    # Check if the destination folder exists, if not, create it
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder, exist_ok=True)
    
    # Regex for filenames in format "frame" followed by numbers (e.g., frame00001)
    frame_pattern = re.compile(r'^frame\d+$')

    # Walk through all subdirectories in the source folder
    for root, dirs, files in os.walk(src_folder):
        # Get the immediate parent folder name
        parent_folder = os.path.basename(os.path.normpath(root))
        
        for file in files:
            # Extract base name and extension
            base_name, ext = os.path.splitext(file)
            
            # Check if the filename matches the "framexxxx" pattern (without extension)
            if frame_pattern.match(base_name):
                # New filename: parent folder name prepended (even if it's src_folder)
                new_base_name = f"{parent_folder}_{base_name}"
            else:
                # For non-frame files: keep original name
                new_base_name = base_name

            # New filename with extension
            new_file_name = new_base_name + ext
            
            # Create paths
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_folder, new_file_name)

            # Copy the file
            shutil.copy2(src_file, dest_file)
            print(f"Copied: {src_file} -> {dest_file}")