# -*- coding: utf-8 -*-
"""
/***************************************************************************
 AssessRFC
                                 A QGIS plugin
 Assess Random Forest Classifier (RFC)
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2018-03-19
        git sha              : $Format:%H$
        copyright            : (C) 2018 by Luis Fernando Chimelo Ruiz
        email                : ruiz.ch@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
import sys
#Insert anaconda path
sys.path.append('C:\ProgramData\Anaconda3\Lib\site-packages')
#QGIS
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qgis.core import *
import numpy as np
import os
import sys
#import RFC

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .assess_rfc_dialog import AssessRFCDialog
from .to_evaluate import is_none, is_defined, exist_file,\
list_is_empty, txt_is_writable,field_is_integer,field_is_real,\
vector_is_readable,is_crs,is_join
import os.path
#Geodata mining
from sklearn import ensemble
from sklearn import metrics
import geopandas as gpd
import time
import rtree

#Get layers QGIS
project = QgsProject.instance()

class AssessRFC:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'AssessRFC_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = AssessRFCDialog()

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&GeoPatterns')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'Random Forest')
        self.toolbar.setObjectName(u'Random Forest')

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('Random Forest', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/assess_rfc/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Random Forest'),
            callback=self.run,
            parent=self.iface.mainWindow())


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Assess RFC'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar
        return 0

    def run(self):
        """Run method that performs all the real work"""
        
        #TabInput visble
        self.dlg.ui.tabWidget.setCurrentIndex(0)
        #Clear GUIs
        self.dlg.ui.comboBoxClassTrain.clear()
        self.dlg.ui.comboBoxClassVal.clear()
        self.dlg.ui.comboBoxCrit.clear()
        self.dlg.ui.comboBoxTrain.clear()
        self.dlg.ui.comboBoxVal.clear()
        self.dlg.ui.lineEditDataSet.clear()
        self.dlg.ui.lineEditOutModel.clear()
        self.dlg.ui.lineEditAssessFile.clear()
        #SetCheckState
        self.dlg.ui.checkBoxApplyModel.setCheckState(False)
        
        
        #enable
        self.dlg.ui.lineEditOutModel.setEnabled(False)
        self.dlg.ui.buttonOutClass.setEnabled(False)
        #Zero progressaBar
        self.dlg.ui.progressBar.setValue(0)
        #select radion button
        self.dlg.ui.radioButtonClass.setChecked(True)
        if self.dlg.ui.radioButtonClass.isChecked():
            print('RadionButton classification checked')
            #clear
            self.dlg.ui.comboBoxCrit.clear()
            #Set criterions Split comboBoxCrit
            self.dlg.ui.comboBoxCrit.addItems(['gini', 'entropy'])
        self.dlg.ui.radioButtonRegress.setChecked(True)
        if self.dlg.ui.radioButtonRegress.isChecked():
            print("RadionButton regression checked")
            #clear
            self.dlg.ui.comboBoxCrit.clear()
            #Set criterions Split comboBoxCrit
            self.dlg.ui.comboBoxCrit.addItems(['mse', 'mae'])
        #Creat Dict with layers names name:layer
        self.dict_layers={'None':None}
        #Insert fields comboBoxs
        self.fields={}
        #Dict with layers names
        dic_layers_qgis =  project.mapLayers()
        for name_layer in dic_layers_qgis.keys():
            #Append name and layer
            self.dict_layers[dic_layers_qgis[name_layer].name()]=dic_layers_qgis[name_layer]
            print (name_layer)
            #assess if is vector
            if dic_layers_qgis[name_layer].type() == 0:
                 #Get field names
                 field_names = [field.name() for field in dic_layers_qgis[name_layer].dataProvider().fields() ]
                 self.fields[dic_layers_qgis[name_layer].name()]=field_names
                 #add name vectors
                 self.dlg.ui.comboBoxTrain.addItem(dic_layers_qgis[name_layer].name())    
                 self.dlg.ui.comboBoxVal.addItem(dic_layers_qgis[name_layer].name())
            else:
                pass

        
        
        #Insert values ComboBoxs                
        if self.dlg.ui.comboBoxTrain.count() == 0:   
            self.dlg.ui.comboBoxTrain.addItem('None')  
            self.dlg.ui.comboBoxClassTrain.addItem('None') 
        else:
            #Get name layer
            names_fields=self.dlg.ui.comboBoxTrain.currentText()           
            #Get and field layer
            self.dlg.ui.comboBoxClassTrain.addItems(self.fields[names_fields])
            
            
        if self.dlg.ui.comboBoxVal.count() == 0:             
            self.dlg.ui.comboBoxVal.addItem('None') 
            self.dlg.ui.comboBoxClassVal.addItem('None')
        else:
            #Get name layer
            names_fields=self.dlg.ui.comboBoxVal.currentText() 
            self.dlg.ui.comboBoxClassVal.addItems(self.fields[names_fields])

        #Connect functions
        self.dlg.ui.buttonSegPath.clicked.connect(self.set_segs_path)
        self.dlg.ui.buttonAssessFile.clicked.connect(self.set_assess_file)
        self.dlg.ui.buttonCancel.clicked.connect(self.cancel_GUI)
        self.dlg.ui.buttonRun.clicked.connect(self.run_classification)
        self.dlg.ui.buttonOutClass.clicked.connect(self.set_classification_path)
        #Connect functions RadionButton
        self.dlg.ui.radioButtonClass.clicked.connect(self.select_model_classification)        
        self.dlg.ui.radioButtonRegress.clicked.connect(self.select_model_regressor)         
        
        #Connect functions value changedspinBox
        self.dlg.ui.spinBoxStartEst.valueChanged.connect(self.value_changed_start_est)
        #self.dlg.ui.spinBoxStepSim.valueChanged.connect(self.value_changed_step_sim)
        self.dlg.ui.spinBoxStartDepth.valueChanged.connect(self.value_changed_start_depth)
        
        #Connect functions valueChanged comboBox
        self.dlg.ui.comboBoxTrain.currentIndexChanged.connect(self.value_changed_train)
        self.dlg.ui.comboBoxVal.currentIndexChanged.connect(self.value_changed_val)
        
        #Connect functions CheckBox
        self.dlg.ui.checkBoxApplyModel.stateChanged.connect(self.state_changed_apply_class)
        
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass
    def select_model_classification(self):
        print('Classification')
        #Clear comboBOx Criterion
        self.dlg.ui.comboBoxCrit.clear()
        #Add criterios
        self.dlg.ui.comboBoxCrit.addItems(['gini', 'entropy'])

    def select_model_regressor(self):
        print('Regressor')
        self.type_model = 'regressor'
        #Clear comboBOx Criterion
        self.dlg.ui.comboBoxCrit.clear()
        #Add criterios
        self.dlg.ui.comboBoxCrit.addItems(['mse', 'mae'])                
        
    def set_classification_path(self):
        #clear lineEdit
        self.dlg.ui.lineEditOutModel.clear()
        #Save shape classification
        out_file = QFileDialog.getSaveFileName(None, self.tr('Save file'), None, " Shapefile (*.shp)")
        #get name and extension file
        name_file= out_file[0].split(os.sep)[-1]
        #Assess extension
        if out_file[-1].split('*')[-1].endswith('.shp)'):
            #insert extension
            if name_file.endswith('.shp'):
                self.dlg.ui.lineEditOutModel.setText(out_file[0])
            else:
                self.dlg.ui.lineEditOutModel.setText(out_file[0]+'.shp')
        else:
            pass
        
    def state_changed_apply_class(self):
        
        state=self.dlg.ui.checkBoxApplyModel.checkState()     
        print ('State apply classification: ',state)
        if state ==2:
            self.dlg.ui.lineEditOutModel.setEnabled(True)
            self.dlg.ui.buttonOutClass.setEnabled(True)
        else: 
            self.dlg.ui.lineEditOutModel.clear()
            self.dlg.ui.lineEditOutModel.setEnabled(False)
            self.dlg.ui.buttonOutClass.setEnabled(False)            

        
    def value_changed_train(self):
        
        #Clear
        self.dlg.ui.comboBoxClassTrain.clear()      
        #Get name layer
        name_layer=self.dlg.ui.comboBoxTrain.currentText() 
        #Get field names
        #field_names = [field.name() for field in self.dict_layers[name_layer].dataProvider().fields() ]
        #Get and field layer
        self.dlg.ui.comboBoxClassTrain.addItems(self.fields[name_layer])    
        print (self.fields[name_layer])
        return 0
    
    def value_changed_val(self):
        #Clear
        self.dlg.ui.comboBoxClassVal.clear()      
        #Get name layer
        name_layer=self.dlg.ui.comboBoxVal.currentText() 
        #Get field names
        #field_names = [field.name() for field in self.dict_layers[name_layer].dataProvider().fields() ]
        #Get and field layer
        self.dlg.ui.comboBoxClassVal.addItems(self.fields[name_layer])    
        print (self.fields[name_layer])
        return 0        
        
    def value_changed_start_est(self):
        #Set minimum valspinBoxEnd
        self.dlg.ui.spinBoxEndEst.setMinimum(self.dlg.ui.spinBoxStartEst.value())
        #Set maximum valspinBoxStep
        self.dlg.ui.spinBoxStepEst.setMaximum(self.dlg.ui.spinBoxEndEst.value())
        return 0
    
    def value_changed_start_depth(self):
        #Set minimum valspinBoxEndSim
        self.dlg.ui.spinBoxEndDepth.setMinimum(self.dlg.ui.spinBoxStartDepth.value())        
        #Set maximum valspinBoxStep
        self.dlg.ui.spinBoxStepDepth.setMaximum(self.dlg.ui.spinBoxEndDepth.value())        
        return 0
    def set_assess_file(self):
        #Clear
        self.dlg.ui.lineEditAssessFile.clear()
        #Save assess file
        assess_file=QFileDialog.getSaveFileName(None, self.tr('Save file'), None, " Text file (*.txt);;Comma-separated values (*.csv)")
        print(assess_file)
        #print (dir(assess_file[-1]))
        #get name and extension file
        name_file=assess_file[0].split(os.sep)[-1]
        #Assess extension
        if assess_file[0]=='':
            self.dlg.ui.lineEditAssessFile.clear()
            
        elif assess_file[-1].split('*')[-1].endswith('.txt)'):

            #insert extension
            if name_file.endswith('.txt'):
                self.dlg.ui.lineEditAssessFile.setText(assess_file[0])
            else:
                self.dlg.ui.lineEditAssessFile.setText(assess_file[0]+'.txt')
        else:
            if name_file.endswith('.csv'):
                self.dlg.ui.lineEditAssessFile.setText(assess_file[0])
            else:
                self.dlg.ui.lineEditAssessFile.setText(assess_file[0]+'.csv')
        
        
        return 0
    def set_segs_path(self):
        #Open Directory
        self.pathSegs = QFileDialog.getExistingDirectory(None, self.tr('Open a folder'), None, QFileDialog.ShowDirsOnly)  
        #Set lineEditPathSegs
        self.dlg.ui.lineEditDataSet.setText(self.pathSegs)
        return 0
    
    def cancel_GUI(self):
        
        #disconnect       
        self.dlg.ui.buttonAssessFile.clicked.disconnect(self.set_assess_file)
        self.dlg.ui.buttonSegPath.clicked.disconnect(self.set_segs_path)
        self.dlg.ui.buttonCancel.clicked.disconnect(self.cancel_GUI)
        self.dlg.ui.buttonRun.clicked.disconnect(self.run_classification)
        self.dlg.ui.comboBoxTrain.currentIndexChanged.disconnect(self.value_changed_train)
        self.dlg.ui.comboBoxVal.currentIndexChanged.disconnect(self.value_changed_val)
        self.dlg.ui.checkBoxApplyModel.stateChanged.disconnect(self.state_changed_apply_class)
        self.dlg.ui.buttonOutClass.clicked.disconnect(self.set_classification_path)

        self.dlg.close()
        
        return 0
    def pontius2011(self,labels_validation,classifier):
            #get class
            labels = np.unique(labels_validation)
            #Get total class
            n_labels=labels.size        
            #print ('n labels: ',n_labels)
            #create matrix 
            sample_matrix = np.zeros((n_labels,n_labels))
            #print sample_matrix
            #print np.count_nonzero(classifier==labels_validation)
    
            #Loop about labels
            for i,l in enumerate(labels):
                #Assess label in classifier
                selec=classifier==l
                #print ( selec.any())
                if selec.any():
                    #Get freqs
                    coords,freqs=np.unique(labels_validation[selec],return_counts=True)
                    #print (coords,freqs)
                    #insert sample_matrix
                    sample_matrix[i,coords-1]=freqs
                    #print( 'l, Freqs: ',l,'---',freqs)
                
            #print (sample_matrix)
            #Sample matrix: samples vs classification
            #sample_matrix=np.histogram2d(classifier,labels_validation,bins=(n_labels,n_labels))[0]
            #coo =np.array([4,5,8,9,11,12,13])-1
            #sample_matrix=sample_matrix[:,coo]
            #sample_matrix=sample_matrix[coo,:]
            #print (sample_matrix.shape)
            #Sum rows sample matrix
            sample_total = np.sum(sample_matrix, axis=1)
            #print ('sum rows: ',sample_total)
            #reshape sample total
            sample_total = sample_total.reshape(n_labels,1)
            #Population total: Image classification or labels validation (random)
            population = np.bincount(labels_validation)
            #Remove zero
            population = population[1:]
            print (population)
            
            #population matrix
            pop_matrix = np.multiply(np.divide(sample_matrix,sample_total,where=sample_total!=0),(population.astype(float)/population.sum()))
            
            #comparison total: Sum rows pop_matrix
            comparison_total = np.sum(pop_matrix, axis=1)
            #reference total: Sum columns pop_matrix
            reference_total = np.sum(pop_matrix, axis=0)
            #overall quantity disagreement
            quantity_disagreement=(abs(reference_total-comparison_total).sum())/2.
            #overall allocation disagreemen
            dig =pop_matrix.diagonal()
            comp_ref=np.dstack((comparison_total-dig,reference_total-dig))
            #allocation_disagreemen=((2*np.min(comp_ref,-1)).sum())/2.
            #proportion correct
            proportion_correct = np.trace(pop_matrix)
            allocation_disagreemen=1-(proportion_correct+quantity_disagreement)
            #print 'Quantity disagreement :',quantity_disagreement
            #print 'Allocation disagreemen :',allocation_disagreemen
            #print 'Proportion correct: ',proportion_correct
            print ('PC: ',proportion_correct, ' DQ: ',quantity_disagreement, 'AD: ',allocation_disagreemen)
            return proportion_correct, quantity_disagreement, allocation_disagreemen, sample_matrix
    
    def RandomForestModel(self,path_train,segs_path,\
                                   path_val,start_est,\
                                   end_est,step_est,start_dp,\
                                   end_dp,step_dp,field_class_train,\
                                   field_class_val,criterion_split,path_assess_file,stateCheckBox,model_path,type_model):
        
            

            #To evaluate vector readable
            if vector_is_readable(path_train,'Error reading the ')==False:
                return 0
            
            #get dataframe training samples
            dft=gpd.read_file(path_train)
        
            #get dataframe validation samples
            dfv=gpd.read_file(path_val)
            #Get CRS
            crsT=dft.crs['init']
            crsV=dfv.crs['init']
            if is_crs (crsT,crsV,'CRS are different - (Training and validation samples)' )==False:
                return 0
            #get names segmentations
            segs_names=[f for f in os.listdir(segs_path)  if f.endswith('.shp')]
            #best parameters
            best_parameters={'Trees':0,'Depth':0}
            #acurcia
            acuracia=0.0
            #segmentations file
            for seg in segs_names: 
                #Set progressa bar            
                self.dlg.ui.progressBar.setValue(1)
                #Selecionar arquivos .shp          
                #f_txt.write(segs_path+os.sep+seg+'\n')
                print (segs_path+os.sep+seg)
                if vector_is_readable(segs_path,'Data set is not readable') == False:
                    return 0
                #Ler segmentacoes
                dfs=gpd.read_file(segs_path+os.sep+seg)
                #To evaluate CRS data set and samples
                crsS = dfs.crs['init']
                if is_crs (crsT,crsS,'CRS are different - (data set)' )==False:
                    return 0

                #create validation samples merge attribute spatial join
                dfjv=gpd.sjoin(dfv,dfs,how="inner", op='intersects')

                #Criar amostras de treinamento, merge attribute spatial join
                dfjt=gpd.sjoin(dft,dfs,how="inner", op='intersects')

                #Get features and remove geometry and id_seg
                if 'id_seg' in dfs.columns:
                    dfs=dfs.drop(columns=['geometry','id_seg'])
                else:
                    dfs=dfs.drop(columns=['geometry'])
                #Get columns names equal dtype=float
                features=dfs.select_dtypes(include='float').columns                
                #Drop NaN validation
                dfjt=dfjt.dropna(subset=features)
                print('dfjv.shape: ',dfjv.shape)
                #To evatualte join
                if is_join(dfjt.shape,'Data set and training samples do not overlap or contains NaN') == False:
                    return 0
                #Drop NaN validation
                dfjv=dfjv.dropna(subset=features)
                print('dfjv.shape: ',dfjv.shape)
                #To evatualte join
                if is_join(dfjv.shape,'Data set and validation samples do not overlap or contains NaN') == False:
                    return 0    

                #create text
                f_txt=open(path_assess_file,'w')
                #To evaluaate if is model type
                if type_model =='classification':
                   #Write 
                   f_txt.write('Dataset;Trees;Depth;PC;QD;QA;kappa'+'\n')
                   
                else:
                    
                    #Write 
                    f_txt.write('Dataset;Trees;Depth;AUC'+'\n') 
                          
                #Avaliar parametros da segmentacao
                for t in range(int(start_est),int(end_est)+int(step_est),int(step_est)):
                    #Set progressbar
                    self.dlg.ui.progressBar.setValue(100/(int(end_est)/t))
                    for md in range(int(start_dp),int(end_dp)+int(step_dp),int(step_dp)):
                        #To evaluaate if is model type
                        if type_model =='classification':
  
                            #criar modelo Random Forest
                            clf = ensemble.RandomForestClassifier( n_estimators =t, max_depth =md, criterion=criterion_split)
                            #Ajustar modelo
                            modelTree = clf.fit(dfjt[features].values, dfjt[field_class_train])
                            #Classificar
                            clas = modelTree.predict(dfjv[features].values)
                            #Calculate kappa
                            kappa = metrics.cohen_kappa_score(dfjv[field_class_val],clas)
                            #Calculate PC
                            pc,qd,qa,matrix=self.pontius2011(dfjv[field_class_val],clas)
                            #print (pc,qd,qa)
                            f_txt.write(seg+';'+str(t)+';'+ str(md)+';'+';'+ str(round(pc,4))+';'+ str(round(qd,4))+';'+ str(round(qa,4))+';'+str(round(kappa,4))+'\n') 
                            #Avaliar a acuracia
                            #print('Acc: '+str(acuracia)+' Pc: '+str(pc))
                            if pc > acuracia:
                                acuracia=pc
                                #Guardar parametros random forest
                                
                                best_parameters['Trees']=t
                                best_parameters['Depth']=md
                                best_parameters['Dataset']=seg
                        else:
                            #criar modelo Random Forest
                            clf = ensemble.RandomForestRegressor( n_estimators =t, max_depth =md, criterion=criterion_split)
                            #Ajustar modelo
                            modelTree = clf.fit(dfjt[features].values, dfjt[field_class_train])
                            #Classificar
                            regress = modelTree.predict(dfjv[features].values)
                            #Calculate kappa
                            auc =metrics.roc_auc_score(dfjv[field_class_val],regress)
                            #print (pc,qd,qa)
                            f_txt.write(seg+';'+str(t)+';'+ str(md)+';'+ str(round(auc,4))+'\n') 
                            #Avaliar a acuracia
                            #print('Acc: '+str(acuracia)+' Pc: '+str(pc))
                            if auc > acuracia:
                                acuracia=auc
                                #Guardar parametros random forest
                                
                                best_parameters['Trees']=t
                                best_parameters['Depth']=md
                                best_parameters['Dataset']=seg
                #Set progressa bar            
                self.dlg.ui.progressBar.setValue(100)  
            #del dataframes
            del(dfs,dfjv,dfjt)   
            #Set progressa bar            
            self.dlg.ui.progressBar.setValue(50) 
            #classificar segmentacao
            f_txt.write('############# Best Parameters #############'+'\n')
            f_txt.write('Data set: '+best_parameters['Dataset']+' - '+'Trees: '+str(best_parameters['Trees'])+ ' - Depth:'+str(best_parameters['Depth'])+'\n')
            ###################### classify best case##############################
            if bool(stateCheckBox) :
                #Ler segmentacoes
                df_dataset=gpd.read_file(segs_path+os.sep+best_parameters['Dataset'])
                #Remove NaN
                df_dataset=df_dataset.dropna(subset=features)
                #create validation samples merge attribute spatial join
                dfjv=gpd.sjoin(dfv,df_dataset,how="inner", op='intersects')
                #Criar amostras de treinamento, merge attribute spatial join
                dfjt=gpd.sjoin(dft,df_dataset,how="inner", op='intersects')
                
                #Apply classification
                if self.dlg.ui.radioButtonClass.isChecked():
                    #criar modelo Random Forest
                    clf = ensemble.RandomForestClassifier( n_estimators =best_parameters['Trees'], max_depth =best_parameters['Depth'],criterion=criterion_split)
                    #Ajustar modelo
                    model = clf.fit(dfjt[features].values, dfjt[field_class_train])
                    #Classificar
                    clas = model.predict(dfjv[features].values)                      
                    #Calculate PC
                    pc,qd,qa,matrix=self.pontius2011(dfjv[field_class_val],clas)
                    #Classificar
                    classification = modelTree.predict(df_dataset[features].values)
                    ##create aux DF classification
                    df_dataset['classes']=classification
                    #output classification
                    df_dataset[['geometry','classes']].to_file( model_path)
                    f_txt.write('############# Confusion Matrix #############'+'\n')
                    f_txt.write(str(matrix)+'\n')
                else:
                    #Create RandomForest Regressor
                    clf = ensemble.RandomForestRegressor( n_estimators =best_parameters['Trees'], max_depth =best_parameters['Depth'],criterion=criterion_split)
                    #Ajustar modelo
                    model = clf.fit(dfjt[features].values, dfjt[field_class_train])
                    #Regressor
                    regress = model.predict(df_dataset[features].values)                      
                    ##create aux DF classification
                    df_dataset['values']=regress          
                    #output classification
                    df_dataset[['geometry','values']].to_file( model_path)
                #Write text
                f_txt.write('############# Features #############'+'\n')
                f_txt.write(str(features)+'\n')
                f_txt.write('############# Features Importances #############'+'\n')
                f_txt.write(str(model.feature_importances_)+'\n')  

                #del
                del(df_dataset,dfjv,dfjt,dfv,dft)
            else:
                pass
    
            print (best_parameters)
            #criar o modelo Random Forest
            #clf = ensemble.RandomForestClassifier( n_estimators =trees, max_depth =max_d,criterion=criterion_split)
            #Ajustar o modelo
            #modelTree = clf.fit(dfj[features].values, dfj[field_class_train])
            #Classificar segmentacao
            #dfs['classify']=modelTree.predict(dfs.values)
            #Calcular a probabilidade e converter para String
            #probs=[str(row) for row in np.round(modelTree.predict_proba(dfs[features].values)*100,2).tolist()]
            #insert geodata frame
            #dfs['probs']=probs
            #Save classify        
            #dfs[['geometry','classify','probs']].to_file(segs_path+os.sep+'class_'+seg)
            #Criar datafame com 
            #Set progressa bar            
            self.dlg.ui.progressBar.setValue(100) 
            f_txt.close()
            
        
      
    def run_classification(self):
        print('Run classification')
        print('Input and output files')

        #Assess lineEdit and combobox training samples
        if is_none(self.dlg.ui.comboBoxTrain.currentText(),'Input: Training samples'):
            return 0       
        #Assess lineEdit and combobox validation samples
        elif is_none(self.dlg.ui.comboBoxVal.currentText(),'Input: Validation samples'):
            return 0
        #Assess lineEdit and combobox data set
        elif is_defined(self.dlg.ui.lineEditDataSet.text(),'Input: data set is not defined'):
            return 0
        #Assess lineEdit and combobox data set
        elif exist_file(self.dlg.ui.lineEditDataSet.text(),'Input: data set is not exist'):
            return 0
        #Assess lineEdit and combobox Output text file
        elif is_defined(self.dlg.ui.lineEditAssessFile.text(),'Output: Text file is not defined'):
            return 0   

        else:
            print('To evaluate')
            #Model path
            model_path=self.dlg.ui.lineEditOutModel.text()
            #Get name and path layer - Input trainining samples
            name_train = self.dlg.ui.comboBoxTrain.currentText()
            path_train=self.dict_layers[name_train].dataProvider().dataSourceUri().split('|')[0]
            
            #Get name field class - training samples
            field_class_train = self.dlg.ui.comboBoxClassTrain.currentText()
            #Type name field (train)
            fields_train=self.dict_layers[name_train].fields()
            type_field_train = fields_train[fields_train.indexFromName(field_class_train)].typeName()

            #Get name and path layer - Input validation samples
            name_val=self.dlg.ui.comboBoxVal.currentText()
            path_val=self.dict_layers[name_val].dataProvider().dataSourceUri().split('|')[0]
            #Get name field class - training samples
            field_class_val = self.dlg.ui.comboBoxClassVal.currentText()
            #Type name field (Val)
            fields_val=self.dict_layers[name_val].fields()
            type_field_val = fields_val[fields_val.indexFromName(field_class_val)].typeName()
            #To evaluate radioButtonClas
            if self.dlg.ui.radioButtonClass.isChecked():
                #Set value type_model
                type_model='classification'
                #Field must be integer
                if field_is_integer(type_field_train,'Input: Training classes field is not integer')==False:
                    return 0
                #Field must be integer
                if field_is_integer(type_field_val,'Input: Validation classes field is not integer')==False:
                    return 0
                print('RadionButton classification checked')
            else:
                #set type model
                type_model='regression'

                #Field must be integer
                if field_is_real(type_field_train,'Input: Training classes field is not real')==False:
                    return 0
                #Field must be integer
                if field_is_real(type_field_val,'Input: Validation classes field is not real')==False:
                    return 0            
            #data set path
            dataset_path=self.dlg.ui.lineEditDataSet.text() 
            #Values n_estimators
            start_est,end_est,step_est = self.dlg.ui.spinBoxStartEst.value(),\
            self.dlg.ui.spinBoxEndEst.value(),self.dlg.ui.spinBoxStepEst.value()
            #values max_depth
            start_dp,end_dp,step_dp=self.dlg.ui.spinBoxStartDepth.value(),\
            self.dlg.ui.spinBoxEndDepth.value(),self.dlg.ui.spinBoxStepDepth.value()
            #Get criterion
            criterion_split = self.dlg.ui.comboBoxCrit.currentText()
            #Assess exist files training samples
            if exist_file(path_train,'Input: Training samples is not exist'):
                return 0
            if vector_is_readable(path_train,'Input: Training samples is not readable') == False:
                return 0
            #Assess exist files training samples
            if exist_file(path_val,'Input: Validation samples is not exist'):
                return 0 
            
            if vector_is_readable(path_val,'Input: Validation samples is not readable') == False:
                return 0
            #Get names files segmentations
            segs_names=[f for f in os.listdir(dataset_path)  if f.endswith('.shp')]
            if  list_is_empty(segs_names, 'Input: Data set folder is empty'):
                return 0  
            
            #Get path assess file
            path_assess_file=self.dlg.ui.lineEditAssessFile.text()
            if is_defined(path_assess_file,'Output: Text file is not defined'):
                return 0
            #Get path assess file
            if txt_is_writable(path_assess_file,'Output: Text file is not exist'):
                return 0
                  
           #To avaliate checkBox Apply Model
            if self.dlg.ui.checkBoxApplyModel.checkState()==2:
                state_checkBox_Class= 'True'
                #Assess   defined vector file
                if is_defined(model_path,'Output: vector file is not defined'):
                    return 0
            else:
                state_checkBox_Class= 'False'
                model_path='/'
            

            #progressbar
            self.dlg.ui.progressBar.setValue(1)
            #Run Assess RFC
            self.RandomForestModel(path_train,dataset_path,\
                    path_val,start_est,end_est,step_est,\
                    start_dp,end_dp,step_dp,field_class_train,\
                    field_class_val,criterion_split,path_assess_file,state_checkBox_Class,model_path,type_model)
            print('Finish')
            #progressbar                
            self.dlg.ui.progressBar.setValue(100)
            self.dlg.ui.progressBar.setTextVisible(True)
            self.dlg.ui.progressBar.setFormat('Finish')
            
   

            
            
