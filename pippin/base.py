from pippin.task import Task


class ConfigBasedExecutable(Task):
    def __init__(self, name, output_dir, base_file, default_assignment, dependencies=None):
        super().__init__(name, output_dir, dependencies=dependencies)
        self.default_assignment = default_assignment
        self.base_file = base_file
        with open(base_file, "r") as f:
            self.base = list(f.read().splitlines())
            self.logger.debug(f"Loaded base file from {self.base_file}")

    def delete_property(self, name, section_start=None, section_end=None):
        self.set_property(name, None, section_start=section_start, section_end=section_end)

    def get_property(self, name, assignment=None):
        """ Get a property from the base file

        Parameters
        ----------
        name : str
        assignment : str, optional

        Returns
        -------
        property : str

        """
        if assignment is None:
            assignment = self.default_assignment

        for line in self.base:
            if line.startswith(name):
                return line.split(assignment)[1]
        return None

    def set_property(self, name, value, section_start=None, section_end=None, assignment=None, only_add=False):
        """ Ensures the property name value pair is set in the base file.
        
        Set value to None to remove a property

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
            Method used to describe setting an attribute. Default determined by class init
        only_add : bool, optional
            Used to add duplicate keys where it would normally replace them.
        """
        if assignment is None:
            assignment = self.default_assignment
        # Want to scan the input files to see if the value exists
        reached_section = section_start is None
        added = False
        desired_line = f"{name}{assignment}{value}"
        for i, line in enumerate(self.base):
            modified_line = line.upper().replace(assignment, " ").strip()
            if reached_section or modified_line.startswith(section_start.upper()):
                reached_section = True
            else:
                continue

            if not only_add and modified_line and modified_line.split()[0] == name.upper():
                # Replace existing option or remove it
                if value is None:
                    self.base[i] = ""
                    self.logger.debug(f"Removing line {i}")
                else:
                    start = line.upper().split(name.upper())[0]
                    self.base[i] = start + desired_line
                    self.logger.debug(f"Setting property on line {i}: {desired_line}")
                added = True
                break

            if value is not None and reached_section and (section_end is not None and line.strip().startswith(section_end)):
                # Option doesn't exist, lets add it
                self.base.insert(i, desired_line)
                added = True
                self.logger.debug(f"Adding new line at location {i}: {desired_line}")
                break
        if not added and value is not None:
            self.base.append(desired_line)
            self.logger.debug(f"Adding to end of file: {desired_line}")
