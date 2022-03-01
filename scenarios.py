import netCDF4 as nc
import torch
from qbo1d import adsolver
from qbo1d.stochastic_forcing import WaveSpectrum

if __name__ == "__main__":

    # global parameters for all scenarios
    t_max = 360 * 108 * 86400
    nsteps = 360 * 108
    nspinup = 360 * 12
    ntot = int(nsteps - nspinup)

    torch.set_default_dtype(torch.float64)


    # scenario 0 (control)
    # --------------------
    solver = adsolver.ADSolver(t_max=t_max, w=3e-4)
    model = WaveSpectrum(solver)
    time = solver.time
    z = solver.z
    u = solver.solve(source_func=model)

    # evaluate source
    s = model.s

    # save to a netcdf file
    file_name = './data/direct/control.nc'
    ds = nc.Dataset(file_name, 'w', format='NETCDF4')

    heights = ds.createDimension('z', z.shape[0])
    times = ds.createDimension('time', ntot)

    heights = ds.createVariable('z', 'f8', ('z'))
    times = ds.createVariable('time', 'f8', ('time'))

    wind = ds.createVariable('u', 'f8', ('time', 'z'))
    source = ds.createVariable('S', 'f8', ('time', 'z'))

    heights[:] = z
    times[:] = time[nspinup:nsteps]

    wind[:, :] = u[nspinup:nsteps, :]
    source[:, :] = s[nspinup:nsteps, :]

    ds.close()


    # scenario I (vertical resolution)
    # --------------------------------
    solver = adsolver.ADSolver(dz=1000, t_max=t_max, w=3e-4)
    model = WaveSpectrum(solver)
    time = solver.time
    z = solver.z
    u = solver.solve(source_func=model)

    # evaluate source
    s = model.s

    # save to a netcdf file
    file_name = './data/direct/scenario1.nc'
    ds = nc.Dataset(file_name, 'w', format='NETCDF4')

    heights = ds.createDimension('z', z.shape[0])
    times = ds.createDimension('time', ntot)

    heights = ds.createVariable('z', 'f8', ('z'))
    times = ds.createVariable('time', 'f8', ('time'))

    wind = ds.createVariable('u', 'f8', ('time', 'z'))
    source = ds.createVariable('S', 'f8', ('time', 'z'))

    heights[:] = z
    times[:] = time[nspinup:nsteps]

    wind[:, :] = u[nspinup:nsteps, :]
    source[:, :] = s[nspinup:nsteps, :]

    ds.close()


    # scenario II (vertical advection)
    # --------------------------------
    solver = adsolver.ADSolver(t_max=t_max, w=5e-4)
    model = WaveSpectrum(solver)
    time = solver.time
    z = solver.z
    u = solver.solve(source_func=model)

    # evaluate source
    s = model.s

    # save to a netcdf file
    file_name = './data/direct/scenario2.nc'
    ds = nc.Dataset(file_name, 'w', format='NETCDF4')

    heights = ds.createDimension('z', z.shape[0])
    times = ds.createDimension('time', ntot)

    heights = ds.createVariable('z', 'f8', ('z'))
    times = ds.createVariable('time', 'f8', ('time'))

    wind = ds.createVariable('u', 'f8', ('time', 'z'))
    source = ds.createVariable('S', 'f8', ('time', 'z'))

    heights[:] = z
    times[:] = time[nspinup:nsteps]

    wind[:, :] = u[nspinup:nsteps, :]
    source[:, :] = s[nspinup:nsteps, :]

    ds.close()


    # scenario III (diffusion coefficient)
    # ------------------------------------
    solver = adsolver.ADSolver(t_max=t_max, w=3e-4, kappa=4e-1)
    model = WaveSpectrum(solver)
    time = solver.time
    z = solver.z
    u = solver.solve(source_func=model)

    # evaluate source
    s = model.s

    # save to a netcdf file
    file_name = './data/direct/scenario3.nc'
    ds = nc.Dataset(file_name, 'w', format='NETCDF4')

    heights = ds.createDimension('z', z.shape[0])
    times = ds.createDimension('time', ntot)

    heights = ds.createVariable('z', 'f8', ('z'))
    times = ds.createVariable('time', 'f8', ('time'))

    wind = ds.createVariable('u', 'f8', ('time', 'z'))
    source = ds.createVariable('S', 'f8', ('time', 'z'))

    heights[:] = z
    times[:] = time[nspinup:nsteps]

    wind[:, :] = u[nspinup:nsteps, :]
    source[:, :] = s[nspinup:nsteps, :]

    ds.close()


    # scenario IV (spectral width)
    # ----------------------------
    solver = adsolver.ADSolver(t_max=t_max, w=3e-4)
    model = WaveSpectrum(solver, cwe=40)
    time = solver.time
    z = solver.z
    u = solver.solve(source_func=model)

    # evaluate source
    s = model.s

    # save to a netcdf file
    file_name = './data/direct/scenario4.nc'
    ds = nc.Dataset(file_name, 'w', format='NETCDF4')

    heights = ds.createDimension('z', z.shape[0])
    times = ds.createDimension('time', ntot)

    heights = ds.createVariable('z', 'f8', ('z'))
    times = ds.createVariable('time', 'f8', ('time'))

    wind = ds.createVariable('u', 'f8', ('time', 'z'))
    source = ds.createVariable('S', 'f8', ('time', 'z'))

    heights[:] = z
    times[:] = time[nspinup:nsteps]

    wind[:, :] = u[nspinup:nsteps, :]
    source[:, :] = s[nspinup:nsteps, :]

    ds.close()


    # scenario V (total flux)
    # -----------------------
    solver = adsolver.ADSolver(t_max=t_max, w=3e-4)
    model = WaveSpectrum(solver, sfe=5e-3)
    time = solver.time
    z = solver.z
    u = solver.solve(source_func=model)

    # evaluate source
    s = model.s

    # save to a netcdf file
    file_name = './data/direct/scenario5.nc'
    ds = nc.Dataset(file_name, 'w', format='NETCDF4')

    heights = ds.createDimension('z', z.shape[0])
    times = ds.createDimension('time', ntot)

    heights = ds.createVariable('z', 'f8', ('z'))
    times = ds.createVariable('time', 'f8', ('time'))

    wind = ds.createVariable('u', 'f8', ('time', 'z'))
    source = ds.createVariable('S', 'f8', ('time', 'z'))

    heights[:] = z
    times[:] = time[nspinup:nsteps]

    wind[:, :] = u[nspinup:nsteps, :]
    source[:, :] = s[nspinup:nsteps, :]

    ds.close()