parser = argparse.ArgumentParser()

parser.add_argument("-a","--arc",type=int  ,help="architecture id" )
parser.add_argument("-c","--cas",type=str  ,help="case name"       )
parser.add_argument("-d","--dat",type=int  ,help="dataset id"      )
parser.add_argument("-e","--epo",type=int  ,help="start from epoch")
parser.add_argument("-g","--gpu",type=int  ,help="gpu id"          )
parser.add_argument("-n","--nep",type=int  ,help="number of epochs")
parser.add_argument("-r","--res",type=float,help="resampling rate" )
parser.add_argument("-s","--sav",type=int  ,help="save every"      )

parser.add_argument("-b","--bat",type=int  ,help="batch size"         )
parser.add_argument("-f","--ftr",type=float,help="training fraction"  )
parser.add_argument("-l","--ler",type=float,help="learning rate"      )
parser.add_argument("-m","--mbs",type=int  ,help="mini batch size"    )
parser.add_argument("-p","--pri",type=int  ,help="print every batches")

parser.add_argument("-w","--wl1",type=float,help="weighted L1 width"  )
parser.add_argument("-k","--eik",type=float,help="factor eikonal loss")

args = parser.parse_args()

if not args.arc:
    args.arc =   0
if not args.cas:
    args.cas = 'run'
if not args.dat:
    args.dat =   0
if not args.epo:
    args.epo =   0
if not args.gpu:
    args.gpu =   0
if not args.nep:
    args.nep = 100
if not args.res:
    args.res =   0.0
if not args.sav:
    args.sav =   5

if not args.bat:
    args.bat = 250
if not args.ftr:
    args.ftr =   0.8
if not args.ler:
    args.ler =   0.01
if not args.mbs:
    args.mbs = args.bat
if not args.pri:
    args.pri =  32

if not args.wl1:
    args.wl1 =  0.0
if not args.eik:
    args.eik =  0.0

networkID     = args.arc
caseName      = args.cas
datasetID     = args.dat
startNew      = not args.epo
oldEpoch      = args.epo
gpuName       = 'cuda:{:1.0f}'.format(args.gpu)
nEpochs       = args.nep
frac_rsmpl    = args.res
resamplYes    = not frac_rsmpl == 0.0
saveEvery     = args.sav

batchSize     = args.bat
frac_train    = args.ftr
leRa          = args.ler
miniBatchSize = args.mbs
printEvery    = args.pri

WL1Width      = args.wl1
useWL1        = WL1Width > 0.0
eikonalFactor = args.eik
useEik        = eikonalFactor > 0.0


deltaNum = 1.0e-10  # finite difference stencil width for gradient estimation
torch.pi = torch.acos(torch.zeros(1)).item() * 2

assert((batchSize%miniBatchSize)==0)

savePath = '../data/network-parameters/'
dataPath = '../data/train-data/'
