import ml_collections

Synapse_train_list=['label0006','label0007' ,'label0009', 'label0010', 'label0021' ,'label0023' ,'label0024','label0026' ,'label0027' ,'label0031', 'label0033' ,'label0034','label0039', 'label0040','label0005', 'label0028', 'label0030', 'label0037']
Synapse_val_list  =['label0001', 'label0002', 'label0003', 'label0004', 'label0008', 'label0022','label0025', 'label0029', 'label0032', 'label0035', 'label0036', 'label0038']

ACDC_train_list=['patient001_frame01', 'patient001_frame12', 'patient004_frame01','patient004_frame15', 'patient005_frame01', 'patient005_frame13','patient006_frame01', 'patient006_frame16', 'patient007_frame01','patient007_frame07', 'patient010_frame01', 'patient010_frame13','patient011_frame01', 'patient011_frame08', 'patient013_frame01','patient013_frame14', 'patient015_frame01', 'patient015_frame10','patient016_frame01', 'patient016_frame12', 'patient018_frame01','patient018_frame10', 'patient019_frame01', 'patient019_frame11','patient020_frame01', 'patient020_frame11', 'patient021_frame01','patient021_frame13', 'patient022_frame01', 'patient022_frame11','patient023_frame01', 'patient023_frame09', 'patient025_frame01','patient025_frame09', 'patient026_frame01', 'patient026_frame12','patient027_frame01', 'patient027_frame11', 'patient028_frame01','patient028_frame09', 'patient029_frame01', 'patient029_frame12','patient030_frame01', 'patient030_frame12', 'patient031_frame01','patient031_frame10', 'patient032_frame01', 'patient032_frame12','patient033_frame01', 'patient033_frame14', 'patient034_frame01','patient034_frame16', 'patient035_frame01', 'patient035_frame11','patient036_frame01', 'patient036_frame12', 'patient037_frame01','patient037_frame12', 'patient038_frame01', 'patient038_frame11','patient039_frame01', 'patient039_frame10', 'patient040_frame01','patient040_frame13', 'patient041_frame01', 'patient041_frame11','patient043_frame01', 'patient043_frame07', 'patient044_frame01','patient044_frame11', 'patient045_frame01', 'patient045_frame13','patient046_frame01', 'patient046_frame10', 'patient047_frame01','patient047_frame09', 'patient050_frame01', 'patient050_frame12','patient051_frame01', 'patient051_frame11', 'patient052_frame01','patient052_frame09', 'patient054_frame01', 'patient054_frame12','patient056_frame01', 'patient056_frame12', 'patient057_frame01','patient057_frame09', 'patient058_frame01', 'patient058_frame14','patient059_frame01', 'patient059_frame09', 'patient060_frame01','patient060_frame14', 'patient061_frame01', 'patient061_frame10','patient062_frame01', 'patient062_frame09', 'patient063_frame01','patient063_frame16', 'patient065_frame01', 'patient065_frame14','patient066_frame01', 'patient066_frame11', 'patient068_frame01','patient068_frame12', 'patient069_frame01', 'patient069_frame12','patient070_frame01', 'patient070_frame10', 'patient071_frame01','patient071_frame09', 'patient072_frame01', 'patient072_frame11','patient073_frame01', 'patient073_frame10', 'patient074_frame01','patient074_frame12', 'patient075_frame01', 'patient075_frame06','patient076_frame01', 'patient076_frame12', 'patient077_frame01','patient077_frame09', 'patient078_frame01', 'patient078_frame09','patient080_frame01', 'patient080_frame10', 'patient082_frame01','patient082_frame07', 'patient083_frame01', 'patient083_frame08','patient084_frame01', 'patient084_frame10', 'patient085_frame01','patient085_frame09', 'patient086_frame01', 'patient086_frame08','patient087_frame01', 'patient087_frame10']
ACDC_val_list  =['patient089_frame01', 'patient089_frame10', 'patient090_frame04','patient090_frame11', 'patient091_frame01', 'patient091_frame09','patient093_frame01', 'patient093_frame14', 'patient094_frame01','patient094_frame07', 'patient096_frame01', 'patient096_frame08','patient097_frame01', 'patient097_frame11', 'patient098_frame01','patient098_frame09', 'patient099_frame01', 'patient099_frame09','patient100_frame01', 'patient100_frame13']

EM_train_list=['train-labels00', 'train-labels01', 'train-labels02', 'train-labels03','train-labels04', 'train-labels05', 'train-labels06', 'train-labels07', 'train-labels08', 'train-labels10', 'train-labels12', 'train-labels14', 'train-labels15', 'train-labels16', 'train-labels17', 'train-labels19', 'train-labels20', 'train-labels23', 'train-labels24', 'train-labels25', 'train-labels26', 'train-labels27', 'train-labels28', 'train-labels29']
EM_val_list  =['train-labels09','train-labels11','train-labels13']

ISIC_train_list=['0000001', '0000002', '0000007', '0000008', '0000009', '0000010', '0000011', '0000017', '0000018', '0000019', '0000021', '0000024', '0000025', '0000026', '0000028', '0000029', '0000030', '0000031', '0000032', '0000034', '0000035', '0000038', '0000039', '0000041', '0000042', '0000044', '0000046', '0000047', '0000049', '0000050', '0000051', '0000054', '0000055', '0000058', '0000059', '0000061', '0000062', '0000063', '0000065', '0000067', '0000068', '0000073', '0000074', '0000075', '0000077', '0000078', '0000085', '0000086', '0000087', '0000091', '0000093', '0000094', '0000095', '0000096', '0000097', '0000103', '0000104', '0000105', '0000108', '0000110', '0000112', '0000114', '0000116', '0000118', '0000119', '0000120', '0000121', '0000123', '0000124', '0000127', '0000128', '0000131', '0000133', '0000134', '0000135', '0000137', '0000139', '0000140', '0000142', '0000143', '0000145', '0000146', '0000148', '0000150', '0000151', '0000152', '0000154', '0000156', '0000157', '0000162', '0000166', '0000167', '0000170', '0000171', '0000173', '0000176', '0000181', '0000182', '0000183', '0000185', '0000186', '0000187', '0000190', '0000191', '0000193', '0000199', '0000204', '0000207', '0000208', '0000209', '0000210', '0000211', '0000214', '0000215', '0000217', '0000218', '0000219', '0000220', '0000224', '0000225', '0000232', '0000235', '0000236', '0000237', '0000239', '0000240', '0000242', '0000243', '0000244', '0000245', '0000249', '0000250', '0000251', '0000255', '0000256', '0000259', '0000260', '0000262', '0000263', '0000264', '0000265', '0000268', '0000269', '0000274', '0000276', '0000277', '0000278', '0000280', '0000285', '0000288', '0000290', '0000293', '0000294', '0000307', '0000313', '0000314', '0000317', '0000321', '0000323', '0000324', '0000326', '0000329', '0000330', '0000331', '0000332', '0000333', '0000337', '0000338', '0000339', '0000341', '0000345', '0000346', '0000347', '0000348', '0000349', '0000350', '0000351', '0000352', '0000353', '0000355', '0000358', '0000359', '0000360', '0000361', '0000363', '0000365', '0000366', '0000369', '0000370', '0000374', '0000376', '0000379', '0000381', '0000382', '0000383', '0000384', '0000385', '0000386', '0000390', '0000391', '0000395', '0000397', '0000403', '0000408', '0000409', '0000410', '0000412', '0000413', '0000416', '0000419', '0000421', '0000425', '0000426', '0000427', '0000431', '0000434', '0000436', '0000439', '0000442', '0000443', '0000445', '0000447', '0000451', '0000453', '0000454', '0000455', '0000457', '0000458', '0000460', '0000461', '0000463', '0000465', '0000467', '0000468', '0000469', '0000471', '0000474', '0000477', '0000478', '0000480', '0000483', '0000485', '0000486', '0000489', '0000491', '0000492', '0000493', '0000495', '0000496', '0000498', '0000500', '0000503', '0000504', '0000505', '0000506', '0000507', '0000513', '0000514', '0000516', '0000521', '0000522', '0000523', '0000528', '0000529', '0000530', '0000531', '0000532', '0000535', '0000536', '0000538', '0000541', '0000542', '0000543', '0000544', '0000545', '0000546', '0000551', '0000552', '0000555', '0000556', '0000882', '0000900', '0000999', '0001102', '0001105', '0001118', '0001119', '0001126', '0001133', '0001134', '0001140', '0001148', '0001152', '0001163', '0001184', '0001187', '0001188', '0001191', '0001212', '0001213', '0001216', '0001247', '0001267', '0001275', '0001296', '0001306', '0001374', '0001385', '0001442', '0002093', '0002206', '0002251', '0002287', '0002353', '0002374', '0002438', '0002453', '0002459', '0002469', '0002476', '0002489', '0002616', '0002647', '0002780', '0002806', '0002836', '0002879', '0002885', '0002975', '0002976', '0003051', '0003174', '0003308', '0003346', '0004110', '0004166', '0004309', '0004715', '0005187', '0005247', '0005548', '0005564', '0005620', '0005639', '0006021', '0006114', '0006326', '0006350', '0006776', '0006800', '0006940', '0006982', '0007038', '0007475', '0007760', '0007788', '0008145', '0008256', '0008280', '0008294', '0008347', '0008396', '0008403', '0008524', '0008541', '0008552', '0008807', '0008879', '0008913', '0008993', '0009160', '0009165', '0009188', '0009252', '0009297', '0009344', '0009430', '0009504', '0009505', '0009533', '0009583', '0009758', '0009800', '0009868', '0009870', '0009871', '0009873', '0009875', '0009877', '0009883', '0009884', '0009888', '0009893', '0009895', '0009896', '0009897', '0009899', '0009900', '0009904', '0009910', '0009911', '0009912', '0009914', '0009919', '0009921', '0009929', '0009933', '0009936', '0009937', '0009938', '0009939', '0009940', '0009944', '0009947', '0009949', '0009950', '0009951', '0009953', '0009961', '0009962', '0009963', '0009964', '0009966', '0009967', '0009968', '0009969', '0009972', '0009973', '0009974', '0009976', '0009979', '0009983', '0009986', '0009987', '0009991', '0009995', '0010000', '0010001', '0010002', '0010003', '0010005', '0010010', '0010014', '0010015', '0010017', '0010019', '0010021', '0010022', '0010024', '0010025', '0010032', '0010035', '0010036', '0010040', '0010042', '0010043', '0010046', '0010051', '0010052', '0010053', '0010054', '0010056', '0010057', '0010060', '0010063', '0010065', '0010067', '0010069', '0010070', '0010071', '0010075', '0010078', '0010079', '0010080', '0010081', '0010083', '0010086', '0010090', '0010093', '0010094', '0010102', '0010104', '0010168', '0010169', '0010174', '0010176', '0010177', '0010178', '0010182', '0010185', '0010189', '0010191', '0010194', '0010212', '0010213', '0010218', '0010219', '0010220', '0010222', '0010223', '0010225', '0010226', '0010227', '0010230', '0010232', '0010233', '0010235', '0010236', '0010237', '0010239', '0010240', '0010241', '0010244', '0010246', '0010247', '0010248', '0010249', '0010252', '0010256', '0010263', '0010264', '0010265', '0010267', '0010320', '0010321', '0010322', '0010323', '0010324', '0010325', '0010327', '0010329', '0010332', '0010333', '0010334', '0010335', '0010339', '0010344', '0010349', '0010350', '0010351', '0010352', '0010356', '0010357', '0010358', '0010361', '0010362', '0010365', '0010367', '0010370', '0010371', '0010380', '0010382', '0010435', '0010436', '0010438', '0010440', '0010441', '0010442', '0010443', '0010445', '0010450', '0010455', '0010457', '0010458', '0010459', '0010461', '0010462', '0010465', '0010466', '0010468', '0010471', '0010472', '0010473', '0010475', '0010476', '0010479', '0010480', '0010481', '0010487', '0010488', '0010490', '0010491', '0010492', '0010493', '0010496', '0010497', '0010554', '0010557', '0010562', '0010566', '0010567', '0010568', '0010569', '0010570', '0010571', '0010573', '0010575', '0010576', '0010577', '0010585', '0010586', '0010589', '0010593', '0010594', '0010595', '0010602', '0010603', '0010605', '0010844', '0010848', '0010849', '0010850', '0010851', '0010852', '0010853', '0010857', '0010858', '0010860', '0010861', '0010862', '0011079', '0011084', '0011085', '0011088', '0011095', '0011097', '0011099', '0011100', '0011109', '0011114', '0011115', '0011116', '0011117', '0011118', '0011120', '0011121', '0011123', '0011124', '0011126', '0011127', '0011128', '0011130', '0011131', '0011135', '0011137', '0011139', '0011140', '0011141', '0011145', '0011146', '0011159', '0011161', '0011163', '0011164', '0011165', '0011166', '0011169', '0011170', '0011200', '0011202', '0011203', '0011207', '0011208', '0011210', '0011211', '0011212', '0011214', '0011215', '0011217', '0011218', '0011223', '0011225', '0011226', '0011228', '0011230', '0011295', '0011296', '0011297', '0011299', '0011301', '0011303', '0011306', '0011315', '0011317', '0011323', '0011324', '0011326', '0011327', '0011328', '0011329', '0011330', '0011331', '0011332', '0011334', '0011341', '0011343', '0011345', '0011346', '0011347', '0011348', '0011352', '0011353', '0011354', '0011356', '0011357', '0011358', '0011360', '0011361', '0011362', '0011372', '0011378', '0011380', '0011382', '0011383', '0011385', '0011387', '0011390', '0011397', '0011400', '0011402']
ISIC_train_list2  =['0000000', '0000004', '0000006', '0000016', '0000045', '0000048', '0000060', '0000079', '0000080', '0000081', '0000082', '0000089', '0000100', '0000102', '0000109', '0000122', '0000130', '0000147', '0000153', '0000155', '0000159', '0000163', '0000175', '0000179', '0000184', '0000189', '0000192', '0000203', '0000205', '0000206', '0000216', '0000221', '0000223', '0000229', '0000247', '0000252', '0000261', '0000275', '0000281', '0000282', '0000283', '0000292', '0000295', '0000297', '0000300', '0000301', '0000303', '0000315', '0000316', '0000322', '0000336', '0000342', '0000344', '0000364', '0000367', '0000372', '0000387', '0000396', '0000415', '0000423', '0000444', '0000452', '0000473', '0000475', '0000481', '0000488', '0000511', '0000517', '0000519', '0000520', '0000524', '0000548', '0000554', '0001106', '0001254', '0001262', '0001286', '0001292', '0001367', '0001372', '0001423', '0001449', '0001742', '0002439', '0002488', '0002948', '0003005', '0004168', '0004985', '0005555', '0005666', '0005787', '0006612', '0006795', '0007087', '0007557', '0008029', '0008236', '0008528', '0008785', '0009599', '0009860', '0009905', '0009909', '0009915', '0009917', '0009925', '0009932', '0009934', '0009935', '0009941', '0009942', '0009960', '0009971', '0009975', '0009978', '0009981', '0010006', '0010029', '0010044', '0010064', '0010066', '0010068', '0010074', '0010087', '0010091', '0010101', '0010105', '0010170', '0010184', '0010186', '0010203', '0010204', '0010205', '0010228', '0010242', '0010251', '0010262', '0010266', '0010317', '0010318', '0010319', '0010330', '0010337', '0010341', '0010342', '0010364', '0010372', '0010439', '0010447', '0010464', '0010467', '0010495', '0010558', '0010572', '0010581', '0010590', '0010864', '0011082', '0011102', '0011105', '0011119', '0011125', '0011136', '0011144', '0011156', '0011157', '0011158', '0011173', '0011199', '0011220', '0011229', '0011304', '0011322', '0011339', '0011350', '0011366', '0011373', '0011393', '0011398']
ISIC_train_list.extend(ISIC_train_list2)
ISIC_val_list = []


def EM_512():
    config = ml_collections.ConfigDict()
    
    config.pretrain = True
    config.deep_supervision = True
    config.train_list = EM_train_list
    config.val_list = EM_val_list
    
    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [512,512]
    config.hyper_parameter.batch_size = 8
    config.hyper_parameter.base_learning_rate = 7e-4
    config.hyper_parameter.model_size = 'Tiny'
    config.hyper_parameter.blocks_num = [3,3,3,3]
    config.hyper_parameter.window_size = [16,16,16,8]
    config.hyper_parameter.val_eval_criterion_alpha = 0.9
    config.hyper_parameter.epochs_num = 2000
    config.hyper_parameter.convolution_stem_down = 8
   
    return config

def ISIC_512():
    config = ml_collections.ConfigDict()
    
    config.pretrain = True
    config.deep_supervision = True
    config.train_list = ISIC_train_list
    config.val_list = ISIC_val_list
    
    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [512,512]
    config.hyper_parameter.batch_size = 16
    config.hyper_parameter.base_learning_rate = 1.3e-4
    config.hyper_parameter.model_size = 'Tiny'
    config.hyper_parameter.blocks_num = [3,3,3,3]
    config.hyper_parameter.window_size = [8,8,16,8]
    config.hyper_parameter.val_eval_criterion_alpha = 0
    config.hyper_parameter.epochs_num = 75
    config.hyper_parameter.convolution_stem_down = 8
   
    return config

def ACDC_224():
    config = ml_collections.ConfigDict()
    
    config.pretrain = True
    config.deep_supervision = False
    config.train_list = ACDC_train_list
    config.val_list = ACDC_val_list
    
    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [224,224]
    config.hyper_parameter.batch_size = 8
    config.hyper_parameter.base_learning_rate = 1e-4
    config.hyper_parameter.model_size = 'Large'
    config.hyper_parameter.blocks_num = [3,3,3,3]
    config.hyper_parameter.window_size = [7,7,14,7]
    config.hyper_parameter.val_eval_criterion_alpha = 0.9
    config.hyper_parameter.epochs_num = 500
    config.hyper_parameter.convolution_stem_down = 4
   
    return config

def Synapse_224():
    config = ml_collections.ConfigDict()
    
    config.pretrain = True
    config.deep_supervision = True
    config.train_list = Synapse_train_list
    config.val_list = Synapse_val_list
    
    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [224,224]
    config.hyper_parameter.batch_size = 16
    config.hyper_parameter.base_learning_rate = 1e-4
    config.hyper_parameter.model_size = 'Base'
    config.hyper_parameter.blocks_num = [3,3,3,3]
    config.hyper_parameter.window_size = [7,7,14,7]
    config.hyper_parameter.val_eval_criterion_alpha = 0.
    config.hyper_parameter.epochs_num = 2700
    config.hyper_parameter.convolution_stem_down = 4
   
    return config
    
def Synapse_320():
    config = ml_collections.ConfigDict()
    
    config.pretrain = True
    config.deep_supervision = True
    config.train_list = Synapse_train_list
    config.val_list = Synapse_val_list
    
    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [320,320]
    config.hyper_parameter.batch_size = 8
    config.hyper_parameter.base_learning_rate = 1e-4
    config.hyper_parameter.model_size = 'Tiny'
    config.hyper_parameter.blocks_num = [3,3,3,3]
    config.hyper_parameter.window_size = [10,10,20,10]
    config.hyper_parameter.val_eval_criterion_alpha = 0.
    config.hyper_parameter.epochs_num = 1300
    config.hyper_parameter.convolution_stem_down = 4
   
    return config
    
CONFIGS = {
    'EM_512':EM_512(),
    'ISIC_512':ISIC_512(),
    'ACDC_224':ACDC_224(),
    'Synapse_224':Synapse_224(),
    'Synapse_320':Synapse_320(),
}



