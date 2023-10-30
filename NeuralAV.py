import tensorflow as tf
from keras import datasets, layers, models, optimizers
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions

data=pd.read_csv('Obfuscated-MalMem2022.csv')

x_0 = data['pslist.nproc'].values
x_1 = data['pslist.nppid'].values
x_2 = data['pslist.avg_threads'].values
x_3 = data['pslist.nprocs64bit'].values
x_4 = data['pslist.avg_handlers'].values
x_5 = data['dlllist.ndlls'].values
x_6 = data['dlllist.avg_dlls_per_proc'].values
x_7 = data['handles.nhandles'].values
x_8 = data['handles.avg_handles_per_proc'].values
x_9 = data['handles.nport'].values
x_10 = data['handles.nfile'].values
x_11 = data['handles.nevent'].values
x_12 = data['handles.ndesktop'].values
x_13 = data['handles.nkey'].values
x_14 = data['handles.nthread'].values
x_15 = data['handles.ndirectory'].values
x_16 = data['handles.nsemaphore'].values
x_17 = data['handles.ntimer'].values
x_18 = data['handles.nsection'].values
x_19 = data['handles.nmutant'].values
x_20 = data['ldrmodules.not_in_load'].values
x_21 = data['ldrmodules.not_in_init'].values
x_22 = data['ldrmodules.not_in_mem'].values
x_23 = data['ldrmodules.not_in_load_avg'].values
x_24 = data['ldrmodules.not_in_init_avg'].values
x_25 = data['ldrmodules.not_in_mem_avg'].values
x_26 = data['malfind.ninjections'].values
x_27 = data['malfind.commitCharge'].values
x_28 = data['malfind.protection'].values
x_29 = data['malfind.uniqueInjections'].values
x_30 = data['psxview.not_in_pslist'].values
x_31 = data['psxview.not_in_eprocess_pool'].values
x_32 = data['psxview.not_in_ethread_pool'].values
x_33 = data['psxview.not_in_pspcid_list'].values
x_34 = data['psxview.not_in_csrss_handles'].values
x_35 = data['psxview.not_in_session'].values
x_36 = data['psxview.not_in_deskthrd'].values
x_37 = data['psxview.not_in_pslist_false_avg'].values
x_38 = data['psxview.not_in_eprocess_pool_false_avg'].values
x_39 = data['psxview.not_in_ethread_pool_false_avg'].values
x_40 = data['psxview.not_in_pspcid_list_false_avg'].values
x_41 = data['psxview.not_in_csrss_handles_false_avg'].values
x_42 = data['psxview.not_in_session_false_avg'].values
x_43 = data['psxview.not_in_deskthrd_false_avg'].values
x_44 = data['modules.nmodules'].values
x_45 = data['svcscan.nservices'].values
x_46 = data['svcscan.kernel_drivers'].values
x_47 = data['svcscan.fs_drivers'].values
x_48 = data['svcscan.process_services'].values
x_49 = data['svcscan.shared_process_services'].values
x_50 = data['svcscan.interactive_process_services'].values
x_51 = data['svcscan.nactive'].values
x_52 = data['callbacks.ncallbacks'].values
x_53 = data['callbacks.nanonymous'].values
x_54 = data['callbacks.ngeneric'].values
x_55 = data['Category'].values

x_data = np.stack((x_0, x_1), axis=1)
y_data = data['Class'].values

color_dict = { 'Benign':'blue', 'Malware':'red' }
plt.scatter(x_data[:,0], x_data[:,1], c=[ color_dict[i] for i in y_data ], cmap="RdBu")
plt.show()

num_dict = { 'Benign': 0, 'Malware': 1}
x_train, x_test, y_train, y_test = train_test_split(x_data, np.array([ num_dict[i] for i in y_data ]), test_size=0.3)

model = models.Sequential()
model.add(layers.Normalization(input_shape = [2,], axis = None))
model.add(layers.Dense(2, activation='sigmoid', activity_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(layers.Dense(1, activation = 'sigmoid', activity_regularizer=tf.keras.regularizers.L2(0.01)))
model.summary()

adam = optimizers.Adam(learning_rate=0.3)

model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=300,
                    validation_data=(x_test, y_test))

plot_decision_regions(x_test, y_test, clf=model, legend=2)
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xlim([0, 300])
plt.ylim([0, 2])
plt.legend(loc='lower right')

train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc = model.evaluate(x_test,  y_test)

print("Training Error:", str(train_acc))
print("Testing Error: ", str(test_acc))