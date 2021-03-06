import yaml
import itertools
from wigner_couple import *
from gen_labels import *
import json


class AcePot():
	def __init__(self,
				elements,
				reference_ens,
				ranks,
				nmax,
				lmax,
				nradbase,
				rcut,
				lmbda,
				**kwargs):
		if kwargs != None:
			self.__dict__.update(kwargs)
		self.E0 = reference_ens
		self.ranks =ranks
		self.elements = elements
		self.deltaSplineBins=0.001
		self.global_ndensity=1
		self.global_FSparams=[1.0, 1.0]
		self.global_rhocut = 100000
		self.global_drhocut = 250
		#assert the same nmax,lmax,nradbase (e.g. same basis) for each bond type
		self.radbasetype = 'ChebExpCos'
		self.global_nmax=nmax
		self.global_lmax=lmax
		assert len(nmax) == len(lmax),'nmax and lmax arrays must be same size'
		self.global_nradbase=nradbase
		if type(rcut) != dict:
			self.global_rcut=rcut
			self.global_lmbda =lmbda
		elif type(rcut) == dict:
			self.rcut = rcut
			self.lmbda = lmbda

		self.set_embeddings()
		self.set_bonds()
		self.set_bond_base()

		lmax_dict = {rank:lv for rank,lv in zip(self.ranks,self.global_lmax)}
		nradmax_dict = {rank:nv for rank,nv in zip(self.ranks,self.global_nmax)}
		mumax_dict={rank:len(self.elements) for rank in self.ranks}
		nulst_1 = [generate_nl(rank,nradmax_dict[rank],lmax_dict[rank],mumax_dict[rank]) for rank in self.ranks]
		print ('descriptors per rank', [len(b) for b in nulst_1])
		nus = [item for sublist in nulst_1 for item in sublist]
		try:
			betas = self.__dict__['betas']
			self.set_funcs(nus,**{'betas':betas})
		except KeyError:
			self.set_funcs(nus)

		return None

	def set_embeddings(self,npoti='FinnisSinclair',FSparams=[1.0,1.0]):#default for linear models in lammps PACE
		#embeddings =dict()#OrderedDict() #{ind:None for ind in range(len(self.elements))}
		embeddings ={ind:None for ind in range(len(self.elements))}
		for elemind in range(len(self.elements)):
			embeddings[elemind] = {'ndensity':self.global_ndensity, 'FS_parameters':FSparams,'npoti':npoti, 'rho_core_cutoff':self.global_rhocut, 'drho_core_cutoff':self.global_drhocut}
		self.embeddings = embeddings

	def set_bonds(self):
		bondinds=range(len(self.elements))
		bond_lsts = [list(b) for b in itertools.product(bondinds,bondinds)]
		self.bondlsts = bond_lsts

	def set_bond_base(self):
		bondstrs = ['[%d, %d]' %(b[0],b[1]) for b in self.bondlsts]
		bonds = {bondstr:None for bondstr in bondstrs}

		#radial basis function expansion coefficients
		#saved in n,l,k shape
		# defaults to orthogonal delta function [g(n,k)] basis of drautz 2019	
		try:
			nradmax = max(self.global_nmax[1:])
		except ValueError:
			nradmax = max(self.global_nmax)
		lmax= max(self.global_lmax)
		nradbase = self.global_nradbase
		crad = np.zeros((nradmax,lmax+1,nradbase),dtype=int)
		for n in range(nradmax):
			for l in range(lmax+1):
				crad[n][l] = np.array([1 if k==n else 0 for k in range(nradbase)]) 

		cnew = np.zeros((nradbase,nradmax,lmax+1))
		for n in range(1,nradmax+1):
			for l in range(lmax+1):
				for k in range(1,nradbase+1):
					cnew[k-1][n-1][l] = crad[n-1][l][k-1]

		for bondlst in self.bondlsts:
			bstr = '[%d, %d]' %(bondlst[0],bondlst[1])
			#bstr = (bondlst[0],bondlst[1])
			try:
				bonds[bstr] = {'nradmax':nradmax, 'lmax':max(self.global_lmax), 'nradbasemax':self.global_nradbase,'radbasename':self.radbasetype,'radparameters':[self.global_lmbda], 'radcoefficients':crad.tolist(), 'prehc':0, 'lambdahc':self.global_lmbda,'rcut':self.global_rcut, 'dcut':0.01, 'rcut_in':0, 'dcut_in':0, 'inner_cutoff_type':'distance'}
			except AttributeError:
				bonds[bstr] = {'nradmax':nradmax, 'lmax':max(self.global_lmax), 'nradbasemax':self.global_nradbase,'radbasename':self.radbasetype,'radparameters':[self.lmbda[bstr]], 'radcoefficients':crad.tolist(), 'prehc':0, 'lambdahc':self.lmbda[bstr],'rcut':self.rcut[bstr], 'dcut':0.01, 'rcut_in':0, 'dcut_in':0, 'inner_cutoff_type':'distance'}
				
		self.bonds = bonds

	def set_funcs(self,nulst,**kwargs):
		permu0 = {b:[] for b in range(len(self.elements))}
		permunu = {b:[] for b in range(len(self.elements))}
		try:
			betas = kwargs['betas']
		except KeyError:
			betas = {nu:1.0 for nu in nulst}
		for nu in nulst:
			mu0,mu,n,l = get_mu_n_l(nu)
			rank = get_mu_nu_rank(nu)
			llst = ['%d']*rank
			lstr = ','.join(b for b in llst) % tuple(l)
			ccs = global_ccs[rank][lstr]
			ms = list(ccs.keys())
			mslsts = [[int(k) for k in m.split(',')] for m in ms]
			msflat= [item for sublist in mslsts for item in sublist]
			if betas[nu] != 0.:
				ccoeffs =  list ( np.array(list(ccs.values())) * betas[nu] )
				permu0[mu0].append({'mu0':mu0,'rank':rank,'ndensity':self.global_ndensity,'num_ms_combs':len(ms),'mus':mu, 'ns':n,'ls':l,'ms_combs':msflat, 'ctildes':ccoeffs})
				permunu[mu0].append(nu)
			elif betas[nu] == 0.:
				print ('Not printing descriptor: %s, coefficient is 0' % nu)
				
		for b in range(len(self.elements)):
			for i in permunu[b]:
				print (b,i)
		for b in range(len(self.elements)):
			print (b,len(permu0[b]))
		self.funcs = permu0

	def write_pot(self,name):
		with open('%s.yace'%name,'w') as writeout:
			e0lst = ['%f']*len(self.elements)
			e0str = ', '.join(b for b in e0lst) % tuple(self.E0)
			elemlst =['%s']*len(self.elements)
			elemstr = ', '.join(b for b in elemlst) % tuple(self.elements)
			writeout.write('elements: [%s] \n' % elemstr)
			writeout.write('E0: [%s] \n' % e0str)
			writeout.write('deltaSplineBins: %f \n' % self.deltaSplineBins)
			writeout.write('embeddings:\n')
			for mu0, embed in self.embeddings.items():
				writeout.write('  %d: ' % mu0)
				ystr = json.dumps(embed) + '\n'
				ystr = ystr.replace('"','')
				writeout.write(ystr)
			writeout.write('bonds:\n')
			bondstrs=['[%d, %d]' %(b[0],b[1]) for b in self.bondlsts]
			for bondstr in bondstrs:
				writeout.write('  %s: ' % bondstr)
				bstr = json.dumps(self.bonds[bondstr]) + '\n'
				bstr = bstr.replace('"','')
				writeout.write(bstr)
			writeout.write('functions:\n')
			for mu0 in range(len(self.elements)):
				writeout.write('  %d:\n'%(mu0))
				mufuncs = self.funcs[mu0]
				for mufunc in mufuncs:
					mufuncstr = '    - ' +json.dumps(mufunc) + '\n'
					mufuncstr = mufuncstr.replace('"','')
					writeout.write(mufuncstr)

elements=['C','H','O']
reference_ens=[0.,0.,0.]
#elements=['Cu']
#reference_ens=[0.]
#nmax=[15,3,2,2,1]
#lmax=[0,2,2,1,1]
#ranks=[1,2,3,4,5]
nmax=[15,2,2,1]
lmax=[0,2,2,1]
ranks=[1,2,2,1]
nradbase=max(nmax)
rcut=7.
lmbda=5.25

Apot = AcePot(elements,reference_ens,ranks,nmax,lmax,nradbase,rcut,lmbda)
Apot.write_pot('coupling_coefficients')
