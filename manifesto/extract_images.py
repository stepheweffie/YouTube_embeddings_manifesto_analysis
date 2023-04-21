import fitz
import os
import dotenv
dotenv.load_dotenv()
pdf = os.environ.get('PDF_FILE_PATH')
# Load the PDF file
pdf_doc = fitz.Document(pdf)

# Create the 'images' directory if it does not exist
if not os.path.exists("../images"):
    os.mkdir("../images")

# Loop through each page in the PDF
for page in pdf_doc:
    # Get a list of all images on the page
    images = page.get_images()
    # Loop through each image and save it
    for i, image in enumerate(images):
        # Get the image data and format
        xref = image[0]
        pix = fitz.Pixmap(pdf_doc, xref)

        # Save the image to a file
        filename = f"images/page{page.number + 1}_image{i + 1}.png"
        pix.save(filename)
        # Remove the image from the page
        pdf_doc.extract_image(xref)

