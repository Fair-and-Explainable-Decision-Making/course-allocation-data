from pdf2image import convert_from_path
import matplotlib.pyplot as plt

# Convert PDFs to images
boxplot_img = convert_from_path("./experiments/figs/reduced_boxplot.pdf")[
    0
]  # First page
hist_img = convert_from_path("./experiments/figs/reduced_hist.pdf")[0]  # First page

# Combine the images in a single figure
fig, axes = plt.subplots(2, 1, figsize=(13, 13))  # Adjust size to match PDF dimensions

# Display the boxplot on the top
axes[0].imshow(boxplot_img)
axes[0].axis("off")  # Hide axes

# Display the histogram on the bottom
axes[1].imshow(hist_img)
axes[1].axis("off")  # Hide axes

# Save as a new PDF
plt.subplots_adjust(wspace=0, hspace=-0.2)  # Remove space between plots
plt.savefig("./experiments/figs/reduced_stacked.jpg", bbox_inches="tight", dpi=300)
plt.savefig(
    "./experiments/figs/redreduced_stacked.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)
