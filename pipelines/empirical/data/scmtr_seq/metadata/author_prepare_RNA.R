library(data.table)
library(purrr)
library(ggplot2)
library(viridis)
library(stringr)
#library(tidyverse)
library(Seurat)

################################################################################  
####################### Creat Seurat object from RNA Matrix ####################
################################################################################  

in.file.dir <- "/Users/yang/data/scMTR_method.dev/Sample_5927_SLX-23063_051623YW_MTR_Def.endo.DNA/RNA"  
dir=list.files(paste0(in.file.dir,"/mtx"),pattern=".mtx",full.names = T)#[c(3,4)]
names(dir) <- str_split(basename(dir),'_',simplify = T) [,6] 
scRNAlist <- list()
for(i in 1:length(dir)){
  print(i)
  counts <- Read10X(data.dir = dir[i]) 
  scRNAlist[[i]] <- CreateSeuratObject(counts, min.cells=3, min.features = 100)
}
scRNA2 <- merge(scRNAlist[[1]], y=c(scRNAlist[[2]],scRNAlist[[3]], scRNAlist[[4]], scRNAlist[[5]],scRNAlist[[6]],scRNAlist[[7]],scRNAlist[[8]]))
scRNA2<-subset(scRNA2, subset = nCount_RNA>=1000 & nFeature_RNA>=500 ) #& nCount_RNA < 2500 & nFeature_RNA < 100000
rm(scRNAlist)
scRNA2@meta.data$sample_bio.id<-str_split(rownames(scRNA2@meta.data),':',simplify = T) [,4]
scRNA2@meta.data$batch<-"Batch1"
scRNA2<-subset(scRNA2, subset = sample_bio.id %in% c("01","02","03","04")) #& nCount_RNA < 2500 & nFeature_RNA < 100000
foo<-scRNA2@meta.data %>% tibble::rownames_to_column("CB") %>% as.data.table
foo[sample_bio.id %in% c("01") ,anno:="Def.endo.D0"]
foo[sample_bio.id %in% c("02") ,anno:="Def.endo.D1"]
foo[sample_bio.id %in% c("03") ,anno:="Def.endo.D2"]
foo[sample_bio.id %in% c("04") ,anno:="Def.endo.D3"]
foo<-foo[,.(CB,anno)] %>% as.data.frame() %>% tibble::column_to_rownames("CB")  %>% .[rownames(scRNA2@meta.data),]
scRNA2@meta.data <-cbind(scRNA2@meta.data,foo)
scRNA2@meta.data$sample_bio.id<-scRNA2@meta.data$foo
scRNA2@meta.data$foo<-NULL
scRNA2@meta.data$CB<-rownames(scRNA2@meta.data)
scRNA2@meta.data$sub.lib<-scRNA2$orig.ident
scRNA2[["percent.mt"]] <- PercentageFeatureSet(scRNA2, pattern = "^MT-")

########################## summary RNA plot ##################################  

pdf(paste0(in.file.dir,"/reads.summary/RNA.seurat.vln.plot.pdf"),width = 15,height= 8)
VlnPlot(scRNA2, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3, group.by = "sample_bio.id")
dev.off()

scRNA2 <- subset(scRNA2, subset = percent.mt <=5 & nCount_RNA < 200000) #& nCount_RNA < 2500 & nFeature_RNA < 100000

pdf(paste0(in.file.dir,"/reads.summary/RNA.seurat.hist.feature.plot.pdf"),width = 8,height= 6)
ggplot(scRNA2@meta.data,aes(log10(nFeature_RNA)))+
  geom_histogram(aes(fill=sample_bio.id),bins =50 )+
  facet_grid(sample_bio.id~.,scales = "free_y")+
  theme_bw()
dev.off()

ggplot(scRNA2@meta.data,aes(x=sample_bio.id,y=(nFeature_RNA)))+
  geom_boxplot( )+
  scale_y_log10()+
  #facet_grid(sample_bio.id~.,scales = "free_y")+
  theme_bw()

ggplot(scRNA2@meta.data,aes(x=sample_bio.id,y=(nCount_RNA)))+
  geom_boxplot( )+
  scale_y_log10()+
  #facet_grid(sample_bio.id~.,scales = "free_y")+
  theme_bw()

pdf(paste0(in.file.dir,"/reads.summary/RNA.seurat.hist.count.plot.pdf"),width = 8,height= 6)
ggplot(scRNA2@meta.data,aes(log10(nCount_RNA)))+
  geom_histogram(aes(fill=sample_bio.id),bins =50 )+
  facet_grid(sample_bio.id~.,scales = "free_y")+
  theme_bw()
dev.off()

saveRDS(scRNA2, file = paste0(in.file.dir,"/scRNA.data.seurat.rds"))
scRNA2<-readRDS(paste0(in.file.dir,"/scRNA.data.seurat.rds"))

plot1 <- FeatureScatter(scRNA2, feature1 = "nCount_RNA", feature2 = "percent.mt",group.by = "sample_bio.id")
plot2 <- FeatureScatter(scRNA2, feature1 = "nCount_RNA", feature2 = "nFeature_RNA",group.by = "sample_bio.id")
pdf(paste0(in.file.dir,"/reads.summary/scatter.count.feature.mt.plot.pdf"),width = 10,height=4)
plot1 + plot2
dev.off()

scRNA2 <- NormalizeData(scRNA2, normalization.method = "LogNormalize", scale.factor = 10000)

scRNA2 <- FindVariableFeatures(scRNA2, selection.method = "vst", nfeatures = 2000)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(scRNA2), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(scRNA2)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

all.genes <- rownames(scRNA2)
scRNA2 <- ScaleData(scRNA2, features = all.genes)
scRNA2 <- RunPCA(scRNA2, features = VariableFeatures(object = scRNA2))

print(scRNA2[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(scRNA2, dims = 1:2, reduction = "pca")

DimPlot(scRNA2, reduction = "pca",dims = c(1,2)) #,group.by= "seurat_clusters"
DimPlot(scRNA2, reduction = "pca",dims = c(1,2),group.by= "sample_bio.id")
DimPlot(scRNA2, reduction = "pca",dims = c(2,3),group.by= "sample_bio.id")
DimPlot(scRNA2, reduction = "pca",dims = c(1,3),group.by= "sample_bio.id")

DimHeatmap(scRNA2, dims = 1, cells = 500, balanced = TRUE)
DimHeatmap(scRNA2, dims = 1:20, balanced = TRUE) #cells = 500, 

scRNA2 <- JackStraw(scRNA2, dims = 20, num.replicate = 100)
scRNA2 <- ScoreJackStraw(scRNA2, dims = 1:20)

dev.off()

pdf(paste0(in.file.dir,"/jackstrawplot.pdf"),width = 5,height = 4)
JackStrawPlot(scRNA2, dims = 1:20)
dev.off()

pdf(paste0(in.file.dir,"/ElbowPlot.pdf"),width = 5,height = 4)

ElbowPlot(scRNA2,ndims = 20)
dev.off()

scRNA2 <- FindNeighbors(scRNA2, dims = 1:7)

i<- 0.5
scRNA2 <- FindClusters(scRNA2, resolution = i)
head(Idents(scRNA2), 5)
scRNA2 <- RunUMAP(scRNA2, dims = 1:7)
pdf(paste0(in.file.dir,"/hE_res.",i,".pdf"),width = 5,height = 4)
#pdf(paste0(in.file.dir,"/h9ESCs_res.",i,".pdf"),width = 5,height = 4)
p1<-DimPlot(scRNA2, reduction = "umap", label = TRUE)+NoLegend()
p2<-DimPlot(scRNA2, reduction = "umap",group.by ="sample_bio.id",label = TRUE)+NoLegend()
print(p1)
print(p2)
#print(p3)
dev.off()

saveRDS(scRNA2, file = paste0(in.file.dir,"/scRNA.data.seurat.processed.rds"))

scRNA2.markers <- FindAllMarkers(scRNA2, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
fwrite(scRNA2.markers %>% as.data.table(),paste0(in.file.dir,"/res.",i,"_markers.tsv"),sep = "\t")

#top.genes
scRNA2.markers %>%
  dplyr::group_by(cluster) %>%
  dplyr::slice_max(n = 3, order_by = avg_log2FC) ->top.genes
png(paste0(in.file.dir,"/res.",i,"_vlnplot.top3.png"),width = 1500,height = 2000,res=150)
p1<-VlnPlot(scRNA2, features =  top.genes$gene, slot = "counts", log = TRUE)
print(p1)
dev.off()

png(paste0(in.file.dir,"/res.",i,"_featureplot.top3.png"),width = 2500,height = 3000,res = 150)
p1<-FeaturePlot(scRNA2, features =top.genes$gene)
print(p1)
dev.off()

scRNA2.markers %>%
  dplyr::group_by(cluster) %>%
  dplyr::top_n(n = 5, wt = avg_log2FC) -> top10
pdf(paste0(in.file.dir,"/res.",i,"_heatmap.top20.pdf"),width = 10,height = 40)
p1<-DoHeatmap(scRNA2, features = top10$gene) + NoLegend()
print(p1)
dev.off()

DotPlot(scRNA2, features = unique(top10$gene), group.by = "sample_bio.id") + RotatedAxis()
