import PTAutoEncoder as PTAE

model_name = "alkane_c5_s_3"
dsfilename = "dscrp-butane-bb-400k"
amodel = PTAE.Autoencoder(input_size=9,encoder_size=[64,256,64],latent_size=3,batch_size=10000,num_epochs=250,learning_rate_start=1e-3,learning_rate_end=1e-5,select_cuda_id=0)
amodel.train_model(dsfilename,omit_first_i_column=1,output_model_name=model_name,interval=10)
