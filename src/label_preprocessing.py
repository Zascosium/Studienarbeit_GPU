import os
import xml.etree.ElementTree as ET
from PIL import Image


def convert_png_to_jpg(images_path):
    """Converts all PNG images in the given directory to JPG format.

    Args:
        images_path: Path to the directory containing PNG images.
    """
    filenames = os.listdir(images_path)
    for filename in filenames:
        if filename.endswith(".png"):
            png_image_path = os.path.join(images_path, filename)
            jpg_image_path = os.path.join(images_path, filename.replace(".png", ".jpg"))

            # Open the PNG image
            with Image.open(png_image_path) as img:
                # Convert PNG to RGB (necessary for saving as JPG)
                rgb_img = img.convert("RGB")
                # Save as JPG
                rgb_img.save(jpg_image_path, "JPEG")

            # Optionally, you can delete the original PNG file after conversion
            os.remove(png_image_path)
    print('converted all .pngs to .jpgs')


def update_xml_files(annotations_path, new_folder_name):
    """Updates XML files with the correct folder and filename, using the XML filename.

    Args:
        annotations_path: Path to the directory containing XML annotation files.
        new_folder_name: The new folder name to replace in <folder>.
    """
    for filename in os.listdir(annotations_path):
        if filename.endswith(".xml"):
            xml_file_path = os.path.join(annotations_path, filename)

            # Read the XML file as bytes
            with open(xml_file_path, "rb") as file:
                xml_content = file.read()

            # Parse the XML content
            root = ET.fromstring(xml_content)

            # Update <folder>
            folder_element = root.find("folder")
            if folder_element is not None:
                folder_element.text = new_folder_name

            # Update <filename> using the XML filename
            filename_element = root.find("filename")
            if filename_element is not None:
                # Use the XML filename and replace .xml with .jpg
                new_filename = filename.replace(".xml", ".jpg")
                filename_element.text = new_filename

            # Save the updated XML file
            tree = ET.ElementTree(root)
            tree.write(xml_file_path, encoding="utf-8", xml_declaration=False)

    print(f"Updated folder and filename in XML")
            

def add_pose_tag_to_xml(directory, default_pose="Unspecified"):
    """
    Fügt ein <pose>-Tag mit einem Standardwert zu allen <object>-Elementen in XML-Dateien hinzu.

    Args:
        directory (str): Pfad zum Verzeichnis mit den XML-Dateien.
        default_pose (str): Der Standardwert für das <pose>-Tag. Standard ist 'Unspecified'.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Iteriere über alle <object>-Tags
            for obj in root.findall("object"):
                # Überprüfen, ob <pose> bereits vorhanden ist
                pose_tag = obj.find("pose")
                if pose_tag is None:
                    # Neues <pose>-Tag hinzufügen
                    pose_tag = ET.SubElement(obj, "pose")
                    pose_tag.text = default_pose

            # Speichern der aktualisierten XML
            tree.write(file_path, encoding="utf-8", xml_declaration=False)


    print(f"<pose>-Tag added to XML")
            
            
def convert_coordinates_to_int(xml_dir):
    """
    Konvertiert die Koordinaten (xmin, ymin, xmax, ymax) in allen XML-Dateien im Verzeichnis von float zu int.

    Args:
        xml_dir (str): Pfad zum Verzeichnis mit den XML-Dateien.
    """
    for filename in os.listdir(xml_dir):
        if filename.endswith(".xml"):
            file_path = os.path.join(xml_dir, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Durchlaufe alle <object>-Tags
            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    for coord in ["xmin", "ymin", "xmax", "ymax"]:
                        value = bndbox.find(coord)
                        if value is not None:
                            # Runden und als int speichern
                            value.text = str(int(float(value.text)))

            # Speichern der geänderten XML-Datei
            tree.write(file_path, encoding="utf-8", xml_declaration=False)


    print(f"converted coordinates to ints")
            

def update_folder_tag(xml_file, split):
    """Update the <folder> tag in the XML file to match the current split (train, validation, test)."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the <folder> element and update its text to the current split
    folder_element = root.find("folder")
    if folder_element is not None:
        folder_element.text = split  # Update folder to train/val/test

    # Save the modified XML back to the file
    tree.write(xml_file)
