import pandas as pd
import openpyxl
import os


def read_files(features, genes):
    # read the features.csv file
    features = pd.read_csv(features, sep=";", usecols=["Unnamed: 0", "Chromosome", "Start", "End", "score"])
    features.rename(columns={"Unnamed: 0": "features"}, inplace=True)

    # read BasepairToGeneMap.tsv file
    genes = pd.read_csv(genes, sep="\t", dtype=str)

    return features, genes


def mapper(features, genes):
    # get the top 20 features
    top_20 = features.loc[:100,:]

    # initialise ExcelWriter object to write to multiple sheets
    with pd.ExcelWriter(os.path.join("..", "output", "mapped_genes.xlsx")) as writer:
        # iterate over the top 20 features
        for index, feature in top_20.iterrows():
            # create a filter for chromsome
            chromosome = (genes.Chromosome.astype(str) == feature.Chromosome.astype(int).astype(str))
            # create a filter for gene start
            start = (genes.Gene_start.astype(int) >= feature.Start.astype(int))
            # create a filter for gene end
            end = (genes.Gene_end.astype(int) <= feature.End.astype(int))
            # apply the filters to filter out the features of interest
            genes_of_interest = genes[(chromosome & start & end)]
            # write these genes to a excel sheet
            genes_of_interest.to_excel(writer, sheet_name=str(int(feature.features)))

    return



if __name__ == '__main__':
    features = os.path.join("..", "output", "features.csv")
    genes = os.path.join("..", "BasepairToGeneMap.tsv")

    features, genes = read_files(features, genes)
    mapper(features, genes)
    print("Done")
