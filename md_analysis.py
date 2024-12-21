import MDAnalysis as mda
import MDAnalysis.analysis.rms
import MDAnalysis.analysis.encore
import warnings
warnings.filterwarnings('ignore')


select_res = '(backbone and (resid 22 173 194 226) and segid NDHA) \
                                           or (backbone and (resid 87) and segid NDHC) or\(backbone and (resid 18 19) and segid NDHH) or \(backbone and (resid 138) and segid NDHG)'


## name_label = ["apo", "MQ", "Quinol"]


if __name__ == "__main__":
    file_path = "/home/muddin/raihan_work/ndh-1/dcd_to_pdb/"
    u_apo = mda.Universe(file_path + "ndh-1msPatriciaSaura.psf", file_path + "run_ndh1ms_last100PatriciaSaura.dcd")


    cluster_apo= MDAnalysis.analysis.encore.cluster(u_apo, selection= select_res)

    universes= [u_apo]
    cluster_set = [cluster_apo]


    for i, cluster_collection in enumerate(cluster_set):
        with open(f"cluster_info_{name_label[i]}", "w") as clust_info:
            clust_info.write('\n')
            clust_info.write(name_label[i]+':\n')
            for cluster in cluster_collection:
                clust_info.write(str(cluster.id) +' ' + '(size: '+ str(cluster.size)+', centroid: '+ str(cluster.centroid)\
                           + ') elements: ' + str(cluster.elements)+'\n')
                time_point=cluster.centroid
                universes[i].trajectory[time_point]
                with mda.Writer(name_label[i]+'_'+str(time_point)+'_frame.pdb') as pdb:
                    pdb.write(universes[i])
