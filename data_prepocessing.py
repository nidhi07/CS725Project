import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#use a converters keyword argument to convert location, event, etc. strings into integers.
def string2num(string):
    return int(string.split(" ")[1])

def preprocess():
	#location where data is stored is the directory data/
	path='data'
	train_df = pd.read_csv(os.path.join(path, 'train.csv'), index_col='id', converters = {'location':string2num})
	#print(train_df.head())
	test = pd.read_csv(os.path.join(path, 'test.csv'), index_col='id', converters = {'location':string2num})
	#print(test.head())
	resource = pd.read_csv(os.path.join(path,'resource_type.csv'), converters = {'resource_type':string2num})
	#print(resource.head())
	severity = pd.read_csv(os.path.join(path,'severity_type.csv'), index_col = 'id',converters = {'severity_type':string2num})
	#print(severity.head())
	events = pd.read_csv(os.path.join(path,'event_type.csv'), converters = {'event_type':string2num})
	#print(events.head())
	log = pd.read_csv(os.path.join(path,'log_feature.csv'), converters = {'log_feature':string2num})
	#print(log.head())
	

	#axis=1 defines that function is to be applied on each row
	#axis=0 means col
	#just concat the 2 lists using the location axis
	loc = pd.concat((train_df[['location']],test[['location']]),axis=0)
	#print(loc.head())
	
	#creating the final df
	#list of attrib and attrib_to_scale
	attrib = []
	attrib_to_scale = []

	#create a empty dataframe with severitys's index - i.e. id
	X = pd.DataFrame(0, index=severity.index, columns = [])
	#print(X.head())

	#add fault severity column from train_df
	X['fault_severity'] = train_df['fault_severity']
	#print(train_df['fault_severity'])
	#print(X.loc[X['fault_severity'].isnull()])
	#print(X.head())

	
	 # Probability attrib should be scaled too
	attrib_to_scale.extend(['loc_prob_{}'.format(i) for i in range(0,3)])
	X['location'] = loc['location']
	attrib.append('location')
	#print(X.head())

	X['sev_type'] = severity.severity_type
	attrib.append('sev_type')
	#print(X.head())

	# location counts
	lc = pd.DataFrame(loc['location'].value_counts()).rename(columns={'location':'loc_count'})
	#print(lc.head())
	X = pd.merge(X, lc, how='left', left_on='location',right_index=True).fillna(0)
	attrib.append('loc_count')
	attrib_to_scale.append('loc_count')
	#print(X.head())

	# binary features for common events
	evtypes = events.event_type.value_counts()
	#print(evtypes)
	events_index = evtypes.index
	#print(events_index)
	#print(events_index.shape)
	#make it binary 0/1 - can have multiple events under one id
	events_type = events.loc[events.event_type.isin(events_index)].groupby(['id','event_type'])['id'].count()
	events_type = events_type.unstack().fillna(0).add_prefix('event_')
	#print(events_type.head())
	#left join
	X = pd.merge(X, events_type, right_index=True, left_index=True, how='left').fillna(0)
	attrib.extend(events_type.columns)
	#print(X.loc[X['event_35'] > 1.0])

	# location based count feature
	X['loc_numbered'] = X.groupby('location')['sev_type'].transform(lambda x: np.arange(x.shape[0])+1)
	X['loc_reverse_numbered'] = X.groupby('location')['sev_type'].transform(lambda x: x.max() + 1 - x)

	# normalize it
	X['normalized_loc_numbered'] = X.groupby('location')['loc_numbered'].transform(lambda x: x/(x.shape[0]+1))
	X['normalized_loc_reverse_numbered'] = X.groupby('location')['loc_reverse_numbered'].transform(lambda x: x/(x.shape[0]+1))

	attrib.append('loc_numbered');
	attrib.append('loc_reverse_numbered');
	attrib.append('normalized_loc_numbered');
	attrib.append('normalized_loc_reverse_numbered');
	attrib_to_scale.append('loc_numbered');
	attrib_to_scale.append('loc_reverse_numbered');
	attrib_to_scale.append('normalized_loc_numbered');
	attrib_to_scale.append('normalized_loc_reverse_numbered');
	#print(X.head())

	# resource types data 
	no_of_resources = pd.DataFrame(resource['id'].value_counts()).rename(columns={'id':'no_of_resources'})
	#print(resource_num)
	X = pd.merge(X, no_of_resources, right_index=True, left_index=True, how='left').fillna(0)
	attrib.append('no_of_resources')
	attrib_to_scale.append('no_of_resources')
	# one-hot common resources
	res_type = resource.resource_type.value_counts()
	resources_index = res_type.index
	res_type = resource.loc[resource.resource_type.isin(resources_index)].groupby(['id','resource_type'])['resource_type'].count()
	res_type = res_type.unstack().fillna(0.).add_prefix('rtype_')
	X = pd.merge(X, res_type, how='left', left_index=True, right_index=True).fillna(0)
	attrib.extend(res_type.columns)
	#print(X.head())

	# log feature data 
	# volume behaves better in log scale - check plot
	log['volume_in_log'] = np.log(log.volume + 1)
		
	log_volume = log.groupby('id')['volume_in_log'].agg(['count','min','mean','max','std','sum']).fillna(0).add_prefix('logvolume_')
	X = pd.merge(X, log_volume, how='left', right_index=True, left_index=True).fillna(0)
	attrib.extend(log_volume.columns)
	attrib_to_scale.extend(log_volume.columns)
	
	##################################################################################################################

	#create final dataframe
	X_train = X.loc[train_df.index, attrib]
	X_test = X.loc[test.index, attrib]
	y_train = X.loc[train_df.index, 'fault_severity']
	y_test = X.loc[test.index,'fault_severity']

	# Scaling
	to_scale = [f for f in attrib if f in attrib_to_scale]
	Y = pd.concat((X_train, X_test),axis=0)
	scaler = StandardScaler()
	scaler.fit(Y[to_scale])
	X_train[to_scale] = scaler.transform(X_train[to_scale])
	X_test[to_scale] = scaler.transform(X_test[to_scale])


	#print(X_train.head())
	#print(pd.DataFrame(y_train).columns)
	#print(X_test.head())
	X_train.to_csv(path_or_buf='/home/vijayalakshmi/Downloads/ML/ML_Project/data/X_train_bk.csv', sep=',', na_rep='', float_format=None, columns=attrib, header=True, index=True, index_label='id', mode='w', encoding=None, compression=None, quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.')

	X_test.to_csv(path_or_buf='/home/vijayalakshmi/Downloads/ML/ML_Project/data/X_test_bk.csv', sep=',', na_rep='', float_format=None, columns=attrib, header=True, index=True, index_label='id', mode='w', encoding=None, compression=None, quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.')

	y_train.to_csv(path='/home/vijayalakshmi/Downloads/ML/ML_Project/data/y_train_bk.csv', sep=',', na_rep='', float_format=None, header=True, index=True, index_label='id', mode='w', encoding=None)

	y_test.to_csv(path='/home/vijayalakshmi/Downloads/ML/ML_Project/data/y_test_bk.csv', sep=',', na_rep='', float_format=None, header=True, index=True, index_label='id', mode='w', encoding=None)

preprocess()
