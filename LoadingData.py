#lavf = llamados atendidos violencia familiar = Family violence attended calls
import pandas as pd

def ReadData():
    
    #Loading data from 2017
    lavf17 = pd.read_csv('llamados-atendidos-violencia-familiar-2017.csv')
    
    #Standarizing the column names acording next years columns
    lavf17.rename(columns = {'llamante_quien_llama':'llamante_descripcion'}, inplace = True)
   
    #Loading data from 2018
    lavf1801 = pd.read_csv('llamados-atendidos-violencia-familiar-201801.csv')
    lavf1802 = pd.read_csv('llamados-atendidos-violencia-familiar-201802.csv')
    lavf1803 = pd.read_csv('llamados-atendidos-violencia-familiar-201803.csv')
    lavf1804 = pd.read_csv('llamados-atendidos-violencia-familiar-201804.csv')
    lavf1805 = pd.read_csv('llamados-atendidos-violencia-familiar-201805.csv')
    lavf1806 = pd.read_csv('llamados-atendidos-violencia-familiar-201806.csv')
    lavf1807 = pd.read_csv('llamados-atendidos-violencia-familiar-201807.csv')
    lavf1808 = pd.read_csv('llamados-atendidos-violencia-familiar-201808.csv')
    lavf1809 = pd.read_csv('llamados-atendidos-violencia-familiar-201809.csv')
    lavf1810 = pd.read_csv('llamados-atendidos-violencia-familiar-201810.csv')
    lavf1811 = pd.read_csv('llamados-atendidos-violencia-familiar-201811.csv')
    lavf1812 = pd.read_csv('llamados-atendidos-violencia-familiar-201812.csv')

    #Loading data from 2019
    lavf1901 = pd.read_csv('llamados-atendidos-violencia-familiar-2019-trimestre-1.csv')
    lavf1902 = pd.read_csv('llamados-atendidos-violencia-familiar-2019-trimestre-2.csv')
    lavf1903 = pd.read_csv('llamados-atendidos-violencia-familiar-2019-trimestre-3.csv')

    #Concatenate 2018 data sets
    lavf18 = pd.concat([lavf1801,lavf1802,lavf1803,lavf1804,lavf1805,lavf1806,lavf1807,lavf1808,lavf1809,lavf1810,lavf1811], axis = 0,  join = "outer" , ignore_index = False)

    #December failed because columns names unmatched. After checking its de same data with different column name, Its standarized.
    # ==> rename(columns = {'llamante_quien_llama':'llamante_descripcion'})
    lavf18.columns = ['caso_id', 'llamante_descripcion', 'llamante_genero','llamante_vinculo_ninios_presentes', 'violencia_tipo','victima_edad','victima_rango_etario', 'victima_genero', 'victima_cantidad','agresor_cantidad', 'agresor_genero','agresor_relacion_victima','llamado_derivacion', 'llamado_fecha_hora', 'llamado_provincia','llamado_provincia_indec_id']

    #Concatenate December
    lavf18 = pd.concat([lavf18,lavf1812], axis = 0,  join = "outer" , ignore_index = False)

    #Same for 2019
    lavf19 = pd.concat([lavf1902,lavf1903], axis = 0,  join = "outer" , ignore_index = False)
    lavf19.columns = ['caso_id', 'llamante_descripcion', 'llamante_genero','llamante_vinculo_ninios_presentes','violencia_tipo','victima_edad','victima_rango_etario','victima_genero','victima_cantidad','agresor_cantidad','agresor_genero','agresor_relacion_victima','llamado_derivacion','llamado_fecha_hora','llamado_provincia','llamado_provincia_indec_id']
    lavf19 = pd.concat([lavf1901,lavf19], axis = 0,  join = "outer" , ignore_index = False)

    #Join all in 1 df
    lavf_all = pd.concat([lavf17,lavf18,lavf19], axis = 0,  join = "outer" , ignore_index = False)

#    return lavf_all

#print('The data sets has been standarized and joined into "lavf_all". Now is ready to start ExplorationDataAnalisis ')
######
#lavf_all.shape = (23420, 16)


#def ClearAnalysis():
    # 1) Null/NaN inputs will be droped
    lavf_all = lavf_all.drop(lavf_all.loc[lavf_all['violencia_tipo'].isnull()].index)
    lavf_all = lavf_all.drop(lavf_all.loc[lavf_all['agresor_cantidad'].isnull()].index)
    lavf_all = lavf_all.drop(lavf_all.loc[lavf_all['victima_rango_etario'].isnull()].index)

    # 2) Columns "llamado_provincia" and "llamado_provincia_indec_id" are the same zone (Buenos Aires City)."Case_id" will be dropped as it doesn't apport any relevant information to this project. 
    lavf_all_new = lavf_all.drop(['llamado_provincia_indec_id','llamado_provincia', 'caso_id'], axis=1)

    # 3) New Features will be created to decompose the time-date column. This new features vare "year","month","day", "full-hour","hour" y "time of the day"
    lavf_all_new['llamado_fecha_hora'] = pd.to_datetime(lavf_all_new['llamado_fecha_hora'])
    lavf_all_new['año'] = lavf_all_new['llamado_fecha_hora'].dt.year
    lavf_all_new['mes'] = lavf_all_new['llamado_fecha_hora'].dt.month
    lavf_all_new['nro_día'] = lavf_all_new['llamado_fecha_hora'].dt.dayofweek
    lavf_all_new.loc[(lavf_all_new['nro_día'] == 0), 'día'] = 'Monday'
    lavf_all_new.loc[(lavf_all_new['nro_día'] == 1), 'día'] = 'Tuesday'
    lavf_all_new.loc[(lavf_all_new['nro_día'] == 2), 'día'] = 'Wednesday'
    lavf_all_new.loc[(lavf_all_new['nro_día'] == 3), 'día'] = 'Thursday'
    lavf_all_new.loc[(lavf_all_new['nro_día'] == 4), 'día'] = 'Friday'
    lavf_all_new.loc[(lavf_all_new['nro_día'] == 5), 'día'] = 'Saturday'
    lavf_all_new.loc[(lavf_all_new['nro_día'] == 6), 'día'] = 'Sunday'
    lavf_all_new = lavf_all_new.drop(['nro_día'], axis=1)
    lavf_all_new['hora'] = lavf_all_new['llamado_fecha_hora'].dt.time
    lavf_all_new['hora_sola'] = lavf_all_new['llamado_fecha_hora'].dt.hour

    lavf_all_new.loc[(lavf_all_new['hora_sola'] >= 0) & (lavf_all_new['hora_sola'] < 6), 'momento_del_dia'] = 'Night'
    lavf_all_new.loc[(lavf_all_new['hora_sola'] >= 6) & (lavf_all_new['hora_sola'] < 12), 'momento_del_dia']   = 'Moorning'
    lavf_all_new.loc[(lavf_all_new['hora_sola'] >= 12) & (lavf_all_new['hora_sola'] < 18), 'momento_del_dia']   = 'Afternoon'
    lavf_all_new.loc[(lavf_all_new['hora_sola'] >= 18) & (lavf_all_new['hora_sola'] < 24), 'momento_del_dia']   = 'Evening'

    # 4) Rows with "No data" and "NN/NA" will be removed. (4958, 19)
    lavf_preparacion1 = lavf_all_new.drop(lavf_all_new[(lavf_all_new['victima_edad'] == 'Sin Datos') 
                                                   | (lavf_all_new['victima_edad'] == '999') 
                                                   | (lavf_all_new['victima_rango_etario'] == 'Sin datos') 
                                                   | (lavf_all_new['victima_rango_etario'] == 'Sin Datos')
                                                   | (lavf_all_new['llamante_genero'] == 'NS/NC')
                                                   | (lavf_all_new['violencia_tipo'] == 'Sin datos')
                                                   | (lavf_all_new['violencia_tipo'] == 'No es un caso de Violencia Familiar')
                                                   | (lavf_all_new['violencia_tipo'] == 'No es un caso de Vio')
                                                   | (lavf_all_new['llamado_derivacion'] == 'No se trata de un caso de violencia familiar')
                                                  ].index, inplace = False)

    lavf_preparacion2 = lavf_preparacion1.drop(lavf_preparacion1[(lavf_preparacion1['llamante_descripcion'] == 'Ns/Nc') 
                                                   | (lavf_preparacion1['llamante_vinculo_ninios_presentes'] == 'Sin datos') 
                                                   | (lavf_preparacion1['llamante_vinculo_ninios_presentes'] == 'NS/NC') 
                                                   | (lavf_preparacion1['victima_genero'] == 'NS/NC')
                                                   | (lavf_preparacion1['agresor_genero'] == 'NS/NC')
                                                   | (lavf_preparacion1['agresor_relacion_victima'] == 'Ns/Nc')   
                                                  ].index, inplace = False)

    convert_dict = {'victima_cantidad': int, 'victima_edad': int}
    lavf_preparacion3 = lavf_preparacion2.astype(convert_dict)

    # 5) Several Upper/Lower case and spelling mistakes will be amend.
    lavf_preparacion3.loc[(lavf_preparacion3['violencia_tipo'] != "Física") & (lavf_preparacion3['violencia_tipo'] != "Psicológica"), 'violencia_tipo'] = 'Otros'
    lavf_preparacion3 = lavf_preparacion3.drop(lavf_preparacion3[(lavf_preparacion3['agresor_cantidad'] == 0) ].index, inplace = False)
    lavf_preparacion3.loc[(lavf_preparacion3['agresor_relacion_victima'] == "Espos@/compañer@ actual"), 'agresor_relacion_victima'] = 'Espos@/compañer@ actual de la víctima'
    lavf_preparacion3.loc[(lavf_preparacion3['agresor_relacion_victima'] == "Ex espos@/pareja"), 'agresor_relacion_victima'] = 'Ex espos@/ ex pareja de la víctima'
    lavf_preparacion3.loc[(lavf_preparacion3['agresor_relacion_victima'] == "Padre"), 'agresor_relacion_victima'] = "Padre de la víctima"
    lavf_preparacion3.loc[(lavf_preparacion3['agresor_relacion_victima'] == "Madre"), 'agresor_relacion_victima'] = "Madre de la víctima"
    lavf_preparacion3.loc[(lavf_preparacion3['agresor_relacion_victima'] == "Herman@"), 'agresor_relacion_victima'] = "Herman@ de la victima"
    lavf_preparacion3.loc[(lavf_preparacion3['agresor_relacion_victima'] == "Abuel@"), 'agresor_relacion_victima'] = "Abuel@ de la victima"
    lavf_preparacion3.loc[(lavf_preparacion3['agresor_relacion_victima'] == "Pareja de la madre/padre"), 'agresor_relacion_victima'] = "Pareja de la madre/padre de la víctima "
    lavf_preparacion3.loc[(lavf_preparacion3['agresor_relacion_victima'] == "Otro no conviviente@"), 'agresor_relacion_victima'] = "Otro no conviviente de la victima"
    lavf_preparacion3.loc[(lavf_preparacion3['agresor_relacion_victima'] == "Otro conviviente"), 'agresor_relacion_victima'] = "Otro conviviente de la victima"
    lavf_preparacion3.loc[(lavf_preparacion3['llamante_vinculo_ninios_presentes'] == "Padrasto"), 'llamante_vinculo_ninios_presentes'] = "Padrastro"
    lavf_preparacion3.loc[(lavf_preparacion3['llamante_descripcion'] == "Familiar"), 'llamante_descripcion'] = "Familiar de la víctima"
    lavf_preparacion3.loc[(lavf_preparacion3['llamante_descripcion'] == "Otro Institucional") | (lavf_preparacion3['llamante_descripcion'] == "Otra Institución "), 'llamante_descripcion'] = "Otra institución"
    lavf_preparacion3.loc[(lavf_preparacion3['llamante_descripcion'] == "Otro Particular"), 'llamante_descripcion'] = "Otro particular"
    print("At first dataset was (23420, 16) of raw information, missing data and typing mistakes. After Exploration Data Analysis, the final size is 20% from original (4265, 19).")  
    return lavf_preparacion3, lavf_all_new



def AgeNoData(lavf_all_new):
    lavf_edades_sindatos = lavf_all_new.loc[lavf_all_new['victima_edad'] == 'Sin Datos']
    lavf_edades_sindatos2 = lavf_edades_sindatos
    lavf_edades_sindatos3 = lavf_edades_sindatos2.drop(lavf_edades_sindatos2[(lavf_edades_sindatos2['llamante_genero'] == 'NS/NC')
                                                   | (lavf_edades_sindatos2['violencia_tipo'] == 'Sin datos')
                                                   | (lavf_edades_sindatos2['violencia_tipo'] == 'No es un caso de Violencia Familiar')
                                                   | (lavf_edades_sindatos2['violencia_tipo'] == 'No es un caso de Vio')
                                                   | (lavf_edades_sindatos2['llamado_derivacion'] == 'No se trata de un caso de violencia familiar')
                                                  ].index, inplace = False)
    lavf_edades_sindatos4 = lavf_edades_sindatos3.drop(lavf_edades_sindatos3[(lavf_edades_sindatos3['llamante_descripcion'] == 'Ns/Nc') 
                                                   | (lavf_edades_sindatos3['llamante_vinculo_ninios_presentes'] == 'Sin datos') 
                                                   | (lavf_edades_sindatos3['llamante_vinculo_ninios_presentes'] == 'NS/NC') 
                                                   | (lavf_edades_sindatos3['victima_genero'] == 'NS/NC')
                                                   | (lavf_edades_sindatos3['agresor_genero'] == 'NS/NC')
                                                   | (lavf_edades_sindatos3['agresor_relacion_victima'] == 'Ns/Nc')   
                                                  ].index, inplace = False)
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['violencia_tipo'] != "Física") 
                          & (lavf_edades_sindatos4['violencia_tipo'] != "Psicológica"), 'violencia_tipo'] = 'Otros'
    lavf_edades_sindatos4 = lavf_edades_sindatos4.drop(lavf_edades_sindatos4[(lavf_edades_sindatos4['agresor_cantidad'] == 0) ].index, inplace = False)
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['agresor_relacion_victima'] == "Espos@/compañer@ actual"), 'agresor_relacion_victima'] = 'Espos@/compañer@ actual de la víctima'
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['agresor_relacion_victima'] == "Ex espos@/pareja"), 'agresor_relacion_victima'] = 'Ex espos@/ ex pareja de la víctima'
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['agresor_relacion_victima'] == "Padre"), 'agresor_relacion_victima'] = "Padre de la víctima"
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['agresor_relacion_victima'] == "Madre"), 'agresor_relacion_victima'] = "Madre de la víctima"
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['agresor_relacion_victima'] == "Herman@"), 'agresor_relacion_victima'] = "Herman@ de la victima"
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['agresor_relacion_victima'] == "Abuel@"), 'agresor_relacion_victima'] = "Abuel@ de la victima"
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['agresor_relacion_victima'] == "Pareja de la madre/padre"), 'agresor_relacion_victima'] = "Pareja de la madre/padre de la víctima "
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['agresor_relacion_victima'] == "Otro no conviviente@"), 'agresor_relacion_victima'] = "Otro no conviviente de la victima"
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['agresor_relacion_victima'] == "Otro conviviente"), 'agresor_relacion_victima'] = "Otro conviviente de la victima"
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['llamante_vinculo_ninios_presentes'] == "Padrasto"), 'llamante_vinculo_ninios_presentes'] = "Padrastro"
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['llamante_descripcion'] == "Familiar"), 'llamante_descripcion'] = "Familiar de la víctima"
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['llamante_descripcion'] == "Otro Institucional"), 'llamante_descripcion'] = "Otra institución" 
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['llamante_descripcion'] == "Otra Institución "), 'llamante_descripcion'] = "Otra institución"
    lavf_edades_sindatos4.loc[(lavf_edades_sindatos4['llamante_descripcion'] == "Otro Particular"), 'llamante_descripcion'] = "Otro particular"
    convert_dict2 = {'victima_cantidad': int} 
    lavf_edades_sindatos4 = lavf_edades_sindatos4.astype(convert_dict2)
    
    return lavf_edades_sindatos4

def ClassificationModels(df):
    lavf_preselectionC = df
    lavf_preselectionC.loc[(lavf_preselectionC['llamado_derivacion'] == "La víctima no aceptó la intervención de Equipos Móviles"), 'llamado_derivacion'] = 'Sin Intervencion'
    lavf_preselectionC.loc[(lavf_preselectionC['llamado_derivacion'] == "No había móviles y/o Equipos para realizar la intervención"), 'llamado_derivacion'] = 'Sin Intervencion'
    lavf_preselectionC.loc[(lavf_preselectionC['llamado_derivacion'] == "Llamante solicitó información y/o orientación"), 'llamado_derivacion'] = "Intervencion"
    lavf_preselectionC.loc[(lavf_preselectionC['llamado_derivacion'] == "No intervino equipo móvil por tratarse de un caso fuera de CABA"), 'llamado_derivacion'] = "Sin Intervencion"
    lavf_preselectionC.loc[(lavf_preselectionC['llamado_derivacion'] == "Se planificó intervención para otro momento"), 'llamado_derivacion'] = "Sin Intervencion"
    lavf_preselectionC.loc[(lavf_preselectionC['llamado_derivacion'] == "Está interviniendo o se deriva a otra institución"), 'llamado_derivacion'] = "Intervencion"
    lavf_preselectionC.loc[(lavf_preselectionC['llamado_derivacion'] == "Intervención Equipos Móviles a donde se encontrara la/s víctima/s"), 'llamado_derivacion'] = "Intervencion"
    
    return lavf_preselectionC
    
    
    
    
    