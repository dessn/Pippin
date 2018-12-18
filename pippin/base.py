import logging
import inspect
import os


class OutputExecutable:
    def __init__(self, output_name):
        self.logger = logging.getLogger("pippin")

        self.output_dir = os.path.abspath(os.path.dirname(inspect.stack()[0][1]) + f"/../output/{output_name}")
        os.makedirs(self.output_dir, exist_ok=True)


class ConfigBasedExecutable(OutputExecutable):
    def __init__(self, base_file, output_name):
        super().__init__(output_name)
        self.logger.debug(f"Loading base file from {self.base_file}")
        with open(base_file, "r") as f:
            self.base = list(f.read().splitlines())
            self.logger.info(f"Loaded base file from {self.base_file}")

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
        desired_line = f"\t{name.upper()} {assignment} {value}"
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
        self.logger.debug(f"Line {i} set to {desired_line}")