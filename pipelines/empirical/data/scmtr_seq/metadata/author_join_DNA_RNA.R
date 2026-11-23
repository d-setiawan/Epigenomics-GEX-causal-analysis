## generate signac files for definitive endoderm 
library(EnsDb.Hsapiens.v86)
library(Signac)
library(Seurat)
library(ggplot2)
library(patchwork)
library(purrr)
library(data.table)
library(stringr)

H3K27me3.color="#E6332A"
H3K27ac.color="#365FAA"
H3K4me1.color="#AFCB31"
H3K4me3.color="#FFE609"
H3K36me3.color="#7CC291"
IgG.color="grey60"


annotation <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevelsStyle(annotation) <- "UCSC"
#genome(annotation) <- "hg38"
seqlengths(annotation)

in.file.dir <- "/Users/yang/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/paired/fragments/fragments"
file.pattern <- "sorted.bed.gz.tbi"
fragment.list<-list.files(in.file.dir,pattern=file.pattern,full.names = T) %>% sub(".tbi","",.)
fragment.list <-fragment.list[grep("D3",fragment.list)]


read.summary <-fread("/Users/yang/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/paired/reads.summary/rna.dna.kept.summary.tsv")
read.summary<-read.summary[lib.type=="RNA"]
###############################################
############# load RNA counts  ################
###############################################


# from 202306.scMTR-step5.RNA.seurat
scRNA<-readRDS("/Users/yang/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/RNA/scRNA.data.seurat.rds")
Assays(scRNA)


# RNA was not filter before to remove the ratio lower than 0.5, as different time point RNA could match hist' s new.cellid
scRNA<-subset(scRNA,subset=CB %in% read.summary[,paste0(sub.lib,"_",cellid)] )

RNAcounts<-scRNA@assays$RNA@counts
colnames(RNAcounts) <-paste0(substring(colnames(RNAcounts),4,12),substring(colnames(RNAcounts),1,2)) 
#Def.endo.D0 Def.endo.D1 Def.endo.D2 Def.endo.D3 
#7719        9408        9061        7679 

###############################################
############# make tile file ##################
###############################################


fragment.list<-list.files(in.file.dir,pattern=file.pattern,full.names = T) %>% sub(".tbi","",.)

for (times in c("D0","D1","D2","D3")){
  fragment.list<-list.files(in.file.dir,pattern=file.pattern,full.names = T) %>% sub(".tbi","",.)
  fragment.list <-fragment.list[grep("D3",fragment.list)]
  frag.path<-fragment.list[1]
  kept.cells<-unique(fread(frag.path)%>% .$V4)
  #7314 8846, 8579, 7098
  kept.cells<-kept.cells[kept.cells %in%colnames(RNAcounts)]
  #7309 ,8844 , 8576, 7098
  for (binsizes in c( 5000,20000)) {
    # binsizes=20000 5000,
    #AggregateTiles:Quantifies fragment counts per cell in fixed-size genome bins across the whole genome, then removes bins with less than a desired minimum number of counts in the bin, then merges adjacent tiles into a single region.
    #GenomeBinMatrix:This function bins the genome and calls FeatureMatrix to construct a bin x cell matrix. 
    
    for (frag.path in fragment.list) {
      #frag.path<-fragment.list[1]
      fragments <- CreateFragmentObject(frag.path,cells = kept.cells)
      counts <- GenomeBinMatrix(
        fragments,
        genome= seqlengths(annotation),
        cells = kept.cells,
        #   min_counts = 0,
        binsize = binsizes,
        verbose = TRUE
      )
      
      saveRDS(counts,paste0(frag.path,"_bin",binsizes/1000,"k.rds"))
    }
    
  }
  
}


########################################################
############# generate signac objects ##################
########################################################
####
# for generate integrated seurat object, generate each modality with data from different time points, integrate all modalities into one
# creates a Seurat object based on the scRNA-seq data

# from 202306.scMTR-step5.RNA.seurat
scRNA<-readRDS("/Users/yang/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/RNA/scRNA.data.seurat.rds")

read.summary <-fread("/Users/yang/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/paired/reads.summary/rna.dna.kept.summary.tsv")
read.summary<-read.summary[lib.type=="RNA"]

scRNA<-subset(scRNA,subset=CB %in% read.summary[,paste0(sub.lib,"_",cellid)] )

scRNA$CB<-paste0(substring(colnames(scRNA),4,12),substring(colnames(scRNA),1,2)) 

in.file.dir <- "/Users/yang/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/paired/fragments/fragments"
file.pattern <- "bin5k.rds"
count.list<-list.files(in.file.dir,pattern=file.pattern,recursive = F,full.names = T) 
i="H3K27me3"
count.list.sub <-count.list[grep(i,count.list)]
kept.cells<-lapply(count.list.sub, function(f) data.table(CB=colnames(readRDS(f)), times=basename(f))) %>% rbindlist() #%>% .[,CB]
kept.cells[duplicated(kept.cells)]
kept.cells<-kept.cells$CB

scRNA<-subset(scRNA,subset= CB %in% kept.cells )
# use CB new cellid as cell id for RNA
counts <- GetAssayData(scRNA,assay = "RNA",slot = "counts") 
colnames(counts) <- scRNA@meta.data$CB 

meta.data<-scRNA@meta.data  %>% tibble::rownames_to_column("cellid") %>% tibble::column_to_rownames("CB")
scRNA <- CreateSeuratObject(counts = counts,meta.data = meta.data)


in.file.dir <- "/Users/wangy/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/paired/fragments/fragments"
file.pattern <- "sorted.bed.gz.tbi"
fragment.list<-list.files(in.file.dir,pattern=file.pattern,full.names = T) %>% sub(".tbi","",.)

for (i in c( "H3K27me3","H3K27ac","H3K4me1","H3K4me3","H3K36me3", "IgG")){
  # i="H3K27me3"
  signac<-c()
  
  fragment.list.sub <-fragment.list[grep(i,fragment.list)]
  count.list.sub <-count.list[grep(i,count.list)]
  for (times in c("D0","D1","D2","D3")){
    #times="D0"
    counts <-readRDS(count.list.sub[grep(times,count.list.sub)])
    signac[[times]] <- CreateChromatinAssay(
      counts = counts,
      annotation = annotation,
      sep = c("-", "-"),
      genome = 'hg38',
      fragments = fragment.list.sub[grep(times,fragment.list.sub)],
      min.cells = 0,
      min.features = 0
    )
  }
  combined <- merge(
    x = signac[["D0"]],
    y = list(signac[["D1"]], signac[["D2"]], signac[["D3"]])  )
  
  ## filter out bins with no detection
  counts <- GetAssayData(combined)   
  genes.percent.expression <- rowMeans(counts>0 )*100   
  genes.filter <- names(genes.percent.expression[genes.percent.expression>0])  
  counts.sub <- counts[genes.filter,]
  
  scRNA[[i]] <- CreateChromatinAssay(
    counts = counts.sub[,colnames(scRNA)],
    annotation = annotation,
    sep = c("-", "-"),
    genome = 'hg38',
    min.cells = 0,
    min.features = 0
  )
  ## add fragments list
  Fragments(scRNA[[i]])<-   combined@fragments
  
}


saveRDS(scRNA,paste0(in.file.dir,"/seurat.rna.hist_bin5k.def.endo.rds"))
