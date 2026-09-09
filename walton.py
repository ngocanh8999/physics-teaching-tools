import matplotlib.pyplot as plt

class WaltonUtils:
  def sp_print(
    self,
    context: str = '', 
    separate: str = '', 
    duplicate: int = 1, 
    start = '', 
    end = '', 
    context_label = '', 
    separate_label = '', 
    get_index=(lambda i: i)
  ):
    def get_context(i):
      label=context_label.format(get_index(i))
      return label + context[len(label):]
    def get_separate(i):
      label=separate_label.format(get_index(i))
      return label + separate[len(label):]

    print(start, end='')
    for i in range(duplicate-1):
      print(get_context(i), end='')
      print(get_separate(i), end='')
    print(get_context(duplicate-1), end=end)

class WaltonMachine:
  def __init__(self, upper_capacitors: int, voltage: float):
    self.utils = WaltonUtils()
    self.upper_capacitors = upper_capacitors
    self.upper_contexts = [voltage if i==0 else 0.0 for i in range(upper_capacitors+1)]
    self.lower_contexts = [0.0 for i in range(upper_capacitors)]
    self.voltage = voltage
    self.context_history = []
    self.write_history()

  def write_history(self):
    voltage_dict = {
      'upper_contexts': self.upper_contexts.copy(),
      'lower_contexts': self.lower_contexts.copy()
    }
    self.context_history.append(voltage_dict)
  
  def update_one(self):
    next_switch = (len(self.context_history) - 1) % 2

    for i in range(self.upper_capacitors):
      mean = (self.upper_contexts[i+next_switch]+self.lower_contexts[i])/2
      self.upper_contexts[i+next_switch] = mean
      self.lower_contexts[i] = mean

    if next_switch == 0:
      self.upper_contexts[-1] = 0 
      self.lower_contexts[0] = self.lower_contexts[1]+self.voltage
    self.upper_contexts[0] = self.upper_contexts[1] + self.voltage

    self.write_history()
  
  def update(self, update_times: int, print_system: bool = True):
    for _ in range(update_times): self.update_one()
    if print_system:
      self.print_system(len(self.context_history) - 1)
  
  def print_system(self, step: int):
    history_len = len(self.context_history)
    if step < -history_len or step >= history_len:
      raise ValueError('Step out of range')
    if step < 0:
      step += history_len
    print(f'System in step {step}', end='\n\n')
    upper_contexts = self.context_history[step]['upper_contexts']
    lower_contexts = self.context_history[step]['lower_contexts']

    self.utils.sp_print(' '*19, '', self.upper_capacitors+1, start=' '*3, end='\n', context_label=' > {:.3e}', get_index=(lambda i: upper_contexts[i]))
    self.utils.sp_print(' '*19, '', self.upper_capacitors, start=' '*19, end='\n', context_label='C{}', get_index=(lambda i:i+1))
    self.utils.sp_print('-'*16, '| |', self.upper_capacitors+1, start='-'*3, end='\n')


    self.utils.sp_print('|', ' '*18, self.upper_capacitors+1, start=' '*10, end='\n')
    self.utils.sp_print('|', ' '*18, self.upper_capacitors+1, start=' '*10, end=f'B{self.upper_capacitors+1}\n', separate_label='B{}', get_index=(lambda i:i+1))
    self.utils.sp_print('|', ' '*18, self.upper_capacitors, start=' '*10 + '|' + '-'*6 + '|+  -|' + '-'*6, end='\n')

    next_switch = step % 2
    if next_switch==0:
      for i in range(9):
        self.utils.sp_print('/', ' '*18, self.upper_capacitors+1, start=' '*(10-i), end='\n') # 26-i

    if next_switch==1:
      for i in range(9):
        self.utils.sp_print('\\', ' '*18, self.upper_capacitors+1, start=' '*(11+i), end='\n')

    D_end = 'D' + str(self.upper_capacitors)
    self.utils.sp_print('|', ' '*18, self.upper_capacitors+1, start=' ', end=D_end + ' '*(18-len(D_end)) + '|\n' , separate_label='D{}')
    self.utils.sp_print('|', ' '*18, self.upper_capacitors+1, start=' ', end=' '*18 + '|\n')

    self.utils.sp_print('-'*16, '| |', self.upper_capacitors, start=' '*12, end=' '*9 + 'Ground\n')
    self.utils.sp_print(' '*19, '', self.upper_capacitors-1, start=' '*28, end='\n', context_label="C{}'", get_index=(lambda i:i+1))
    self.utils.sp_print(' '*19, '', self.upper_capacitors, start=' '*12, end='\n', context_label=' > {:.3e}', get_index=(lambda i: lower_contexts[i]))

  def get_sum_voltage(self, step: int):
    if step < 0 or step >= len(self.context_history):
      raise ValueError('Step out of range')
    first_context = self.context_history[step]['upper_contexts'][0]
    last_context = self.context_history[step]['upper_contexts'][-1]
    return first_context - last_context
  
  def get_capacitor_voltage(self, step: int, capacitor: int, is_lower: bool = False):
    if step < 0 or step >= len(self.context_history):
      raise ValueError('Step out of range')
    if capacitor < 0 or capacitor > self.upper_capacitors - int(is_lower):
      raise ValueError('Capacitor out of range')
    first_context = self.context_history[step]['upper_contexts'][capacitor-1]
    second_context = self.context_history[step]['upper_contexts'][capacitor]
    return first_context - second_context

  def plot_history(
    self, 
    step_range: tuple[int, int] = [], 
    only_odd_step: bool = True, 
    have_max_voltage: bool = True,
    have_sum_voltage: bool = True,
    upper_capacitors: list[int] = [],
    lower_capacitors: list[int] = [],
    all_capacitors: bool = False,
  ):
    # Filter steps index
    history_len = len(self.context_history)
    if len(step_range) < 2: step_range = [0, history_len]
    if step_range[0] > history_len or step_range[0] < 0:
      raise ValueError('Your steps out of range')
    if step_range[0] >= step_range[1]:
      raise ValueError('Upper step bound should greater than lower bound')
    if step_range[1] > history_len:
      print(f'Warning: current max step is {history_len-1}')
      step_range[1] = history_len

    steps = []
    for i in range(step_range[0], step_range[1]):
      if only_odd_step:
        if i%2==1: steps.append(i)
        continue
      steps.append(i)

    plt.xlabel('Step')
    plt.ylabel('Voltage')

    if have_sum_voltage:
      plt.plot(
        steps,
        [self.get_sum_voltage(i) for i in steps],
        label='Sum voltage'
      )

    if have_max_voltage:
      plt.plot(
        [0, history_len], 
        [self.voltage * self.upper_capacitors for _ in range(2)],
        '-.',
        label='Max voltage'
      )
    
    upper_capacitors = range(self.upper_capacitors) if all_capacitors else upper_capacitors
    for capacitor in upper_capacitors:
      plt.plot(
        steps,
        [self.get_capacitor_voltage(i, capacitor) for i in steps],
        label=f'C{capacitor} voltage',
      )
    lower_capacitors = range(self.upper_capacitors-1) if all_capacitors else lower_capacitors
    for capacitor in lower_capacitors:
      plt.plot(
        steps,
        [self.get_capacitor_voltage(i, capacitor, True) for i in steps],
        '--',
        label=f"C{capacitor}' voltage",
      )
    
    plt.legend(loc='best')