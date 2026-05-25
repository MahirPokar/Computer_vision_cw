data:
    @echo "Dataset already downloaded via Colab"

bow: 
    python traditional_cv/bow_grid.py --detector sift
    python traditional_cv/bow_grid.py --detector orb

figs: bow
    python results/make_figures.py
