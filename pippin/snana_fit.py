import inspect
import os


class SNANALightCurveFit:
    def __init__(self, base="base_des.nml", fitopts="base_des.fitopts"):
        self.data_dir = os.path.dirname(inspect.stack()[0][1]) + "/data_files/"
        self.base_file = base
        self.fitopts_file = fitopts

        with open(self.data_dir + base, "r") as f:
            self.base = list(f.read().splitlines())
        with open(self.data_dir + fitopts, "r") as f:
            self.fitopts = list(f.read().splitlines())

    def set_property(self, name, value, section_start=None, section_end=None, assignment="="):
        """ Ensures the property name value pair is set in the base file.
        
        Parameters
        ----------
        name : str
            The name of the property. Case insensitive, will be cast to upper.
        value : object
            The value to use. Object will be cast to string. For strings, include single quotes.
        section_start : str, optional
            What section to add the parameter too. Generally "&SNLCINP" or "&FITINP"
        section_end : str, optional
            What ends the section. Generally "&END"
        assignment : str, optional
            Method used to describe setting an attribute. Normally "=", but might need to 
            be ":" for some cases.
        """
        # Want to scan the input files to see if the value exists
        reached_section = section_start is None
        added = False
        desired_line = f"\t{name.upper()} = {value}"
        for i, line in enumerate(self.base):
            if reached_section or line.strip().startswith(section_start):
                reached_section = True
            else:
                continue

            if line.strip().upper().startswith(name.upper()):
                # Replace existing option
                self.base[i] = desired_line
                added = True
                break

            if reached_section and (section_end is not None and line.strip().startswith(section_end)):
                # Option doesn't exist, lets add it
                self.base.insert(i, desired_line)
                added = True
                break
        if not added:
            self.base.append(desired_line)

    def set_snlcinp(self, name, value):
        """ Ensures the property name value pair is set in the SNLCINP section.

        Parameters
        ----------
        name : str
            The name of the property. Case insensitive, will be cast to upper.
        value : object
            The value to use. Object will be cast to string. For strings, include single quotes.
        """
        self.set_property(name, value, section_start="&SNLCINP", section_end="&END")

    def set_fitinp(self, name, value):
        """ Ensures the property name value pair is set in the FITINP section.

        Parameters
        ----------
        name : str
            The name of the property. Case insensitive, will be cast to upper.
        value : object
            The value to use. Object will be cast to string. For strings, include single quotes.
        """
        self.set_property(name, value, section_start="&FITINP", section_end="&END")

if __name__ == "__main__":
    s = SNANALightCurveFit()

    s.set_snlcinp("CUTWIN_NBAND_THRESH", 1000)
    s.set_snlcinp("HELLO", "'human'")
    s.set_fitinp("FITWIN_PROB", "0.05, 1.01")
    s.set_fitinp("GOODBYE", -1)
    s.set_property("BATCH_INFO", "sbatch  $SBATCH_TEMPLATES/SBATCH_sandyb.TEMPLATE  96", assignment=":")
    for line in s.base:
        print(line)
