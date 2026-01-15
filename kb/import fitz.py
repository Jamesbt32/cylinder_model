import fitz
doc = fitz.open("C:\\Users\\james\\OneDrive\\Documents\\Python\\knowledge\\8000014609_03.pdf")
for pnum, page in enumerate(doc, 1):
    imgs = page.get_images(full=True)
    print(f"Page {pnum}: {len(imgs)} images")
    