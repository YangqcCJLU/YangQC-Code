import scipy.io
import matplotlib.pyplot as plt

raw = scipy.io.loadmat(".../raw1.mat")['raw']
mean_spec = scipy.io.loadmat(".../mean_spec.mat")['mean_spec']
std_all = scipy.io.loadmat(".../std_all.mat")['std_all']
wavelengths = scipy.io.loadmat(".../wavelengths.mat")['wavelengths']
wavelengths = wavelengths[49:250].reshape(-1)


# ============== raw spectra ==============
plt.figure()
for i in range(raw.shape[0]):
    plt.plot(wavelengths, raw[i, :], linewidth=0.8)
plt.savefig("raw.tif", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
# ============== mean spectra ==============
plt.figure()
plt.plot(wavelengths, mean_spec[0, :], color='g', label='none')
plt.plot(wavelengths, mean_spec[1, :], color='b', label='procymidone')
plt.plot(wavelengths, mean_spec[2, :], color='c', label='oxytetracycline')
plt.plot(wavelengths, mean_spec[3, :], color='y', label='indoleacetic acid')
plt.plot(wavelengths, mean_spec[4, :], color='r', label='gibberellin')
plt.fill_between(wavelengths, mean_spec[0, :]-std_all[0, :], mean_spec[0, :]+std_all[0, :], color='g', alpha=0.1)
plt.fill_between(wavelengths, mean_spec[1, :]-std_all[1, :], mean_spec[1, :]+std_all[1, :], color='b', alpha=0.1)
plt.fill_between(wavelengths, mean_spec[2, :]-std_all[2, :], mean_spec[2, :]+std_all[2, :], color='c', alpha=0.1)
plt.fill_between(wavelengths, mean_spec[3, :]-std_all[3, :], mean_spec[3, :]+std_all[3, :], color='y', alpha=0.1)
plt.fill_between(wavelengths, mean_spec[4, :]-std_all[4, :], mean_spec[4, :]+std_all[4, :], color='r', alpha=0.1)
plt.xlabel('Wavelength(nm)', fontsize=16)
plt.ylabel('Reflectance', fontsize=16)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.xlim([480, 927])
plt.legend()
legend = plt.legend()
for text in legend.get_texts():
    text.set_fontsize(14)
plt.savefig("mean.tif", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
