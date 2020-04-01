import sys

import parse as parser
import re


def pc_sep_set_type(s):
    if len(s) == 2:
        return []
    return list(map(int, s[1:-1].split(', ')))


def seq_spaced(s):
    return list(map(int, s.split()))


class LogParser:

    def __init__(self, log_path, p_value, labels):
        self._log_path = log_path
        self._p_value = p_value
        self._num_to_label = labels
        self._label_to_num = {label: i for i, label in enumerate(labels)}
        log_file = open(log_path, 'r')
        self._sep_sets = {}
        self._orientations = {}
        self._pc_sep_set_pattern = parser.compile("x = {:d}, y = {:d}, S = {:sep_set_type} : pval = {:g}",
                                                  extra_types=dict(sep_set_type=pc_sep_set_type))
        self._fci_sep_set_pattern = parser.compile("x= {:d}  y= {:d}  S= {:sep_set_type} : pval = {:g}",
                                                   extra_types=dict(sep_set_type=seq_spaced))
        self._rule_pattern = parser.compile("Rule {:d}")
        self._rule_1_pattern = parser.compile("Orient: {:d} *-> {:d} o-* {:seq_type} "
                                              "as: {:d} -> {:seq_type}",
                                              extra_types=dict(seq_type=seq_spaced))
        self._rule_2_pattern = parser.compile("Orient: {:d} -> {:seq_type} *-> {:d} "
                                              "or {:d} *-> {:seq_type} -> {:d} "
                                              "with {:d} *-o {:d} "
                                              "as: {:d} *-> {:d}",
                                              extra_types=dict(seq_type=seq_spaced))

        self._rule_4a_pattern = parser.compile("There is a discriminating path between {:d} and {:d} for {:d} ,"
                                               "and {:d} is not in Sepset of {:d} and {:d} . "
                                               "Orient: {:d} <-> {:d} <-> {:d}",
                                               extra_types=dict(seq_type=seq_spaced))
        self._collider_pattern = parser.compile("{:d} *-> {:d} <-* {:d}")
        self._parse(log_file)
        log_file.close()

    def _parse_pc_skeleton(self, log_file):
        for line in log_file:
            line = line.strip()
            if len(line) == 0:
                break
            if line[0] != 'x':
                continue
            res = self._pc_sep_set_pattern.parse(line)
            p = res[3]
            if p < self._p_value:
                continue
            x = res[0]
            y = res[1]
            Z = res[2]
            self._sep_sets[(x, y)] = (p, Z)
            self._sep_sets[(y, x)] = (p, Z)

    # actually doesn't parse, only skip
    def _parse_building_pdsep(self, log_file):
        for line in log_file:
            line = line.strip()
            if line == "Compute collider:":
                break

    # actually doesn't parse, only skip
    def _parse_pc_colliders(self, log_file):
        for line in log_file:
            line = line.strip()
            if line.startswith('Possible D-Sep of'):
                break

    def _parse_fci_skeleton(self, log_file):
        for line in log_file:
            line = line.strip()
            if len(line) == 0:
                continue
            if line == 'Direct egdes:':
                break
            if line[0] != 'x':
                continue
            res = self._fci_sep_set_pattern.parse(line)
            p = res[3]
            if p < self._p_value:
                continue
            # Alarm! In FCI skeleton log it numerates from 1, not 0.
            x = res[0] - 1
            y = res[1] - 1
            Z = map(lambda x: x - 1, res[2])
            self._sep_sets[(x, y)] = (p, Z)
            self._sep_sets[(y, x)] = (p, Z)

    # Alarm! In FCI skeleton log it numerates from 1, not 0.
    def _orient_colliders(self, log_file):
        for line in log_file:
            line = line.strip()
            if line.startswith("Rule "):
                return line  # bad design, but ftell() and fseek don't work even in while loop
            res = self._collider_pattern.parse(line)
            if res is not None:
                sepsets = log_file.readline()
                # sepsets = parser.compile('Sxz= {:seq_type} and Szx= {:seq_type}',
                #                          extra_types=dict(seq_type=seq_spaced))
                x = res[0] - 1
                z = res[1] - 1
                y = res[2] - 1
                self._orientations[(x, z)] = 'Collider: ' + line + '\n' + sepsets
                self._orientations[(y, z)] = 'Collider: ' + line + '\n' + sepsets

    # Alarm! In FCI skeleton log it numerates from 1, not 0.
    def _handle_rule(self, n_rule, description):
        # reason = 'Rule {}: '.format(n_rule) + description
        reason = description
        if n_rule == 1:
            res = self._rule_1_pattern.parse(description)
            y = res[1] - 1
            Z = list(map(lambda x: x - 1, res[2]))
            for z in Z:
                self._orientations[(y, z)] = reason
                self._orientations[(z, y)] = reason
        if n_rule == 2:
            res = self._rule_2_pattern.parse(description)
            y = res[-2] - 1
            z = res[-1] - 1
            self._orientations[(y, z)] = reason
        if n_rule == 4:
            res = self._rule_4a_pattern.parse(description)
            x = res[-3] - 1
            y = res[-2] - 1
            z = res[-1] - 1
            self._orientations[(x, y)] = reason
            self._orientations[(y, x)] = reason
            self._orientations[(y, z)] = reason
            self._orientations[(z, y)] = reason

    def _orient_with_rules(self, log_file, first_rule=None):
        rule = first_rule
        n_rule = self._rule_pattern.parse(rule)[0]
        description = log_file.readline().strip()
        self._handle_rule(n_rule, description)

        for line in log_file:
            line = line.strip()
            if len(line) == 0:
                continue
            n_rule = self._rule_pattern.parse(line)[0]
            description = log_file.readline().strip()
            self._handle_rule(n_rule, description)

    def _parse_directions(self, log_file):
        first_rule = self._orient_colliders(log_file)
        self._orient_with_rules(log_file, first_rule)

    def _parse(self, log_file):
        log_file.readline()  # Compute Skeleton
        log_file.readline()  # ================
        log_file.readline()  # Casting arguments...

        self._parse_pc_skeleton(log_file)
        log_file.readline()  # Compute PDSEP
        log_file.readline()  # =============
        self._parse_building_pdsep(log_file)
        self._parse_pc_colliders(log_file)
        self._parse_fci_skeleton(log_file)

        log_file.readline()  # =============
        log_file.readline()  # Using rules: 1 2 3 4 5 6 7 8 9 10
        log_file.readline()  # Compute collider:

        self._parse_directions(log_file)

    def _get_num_and_label(self, x):
        if isinstance(x, str):
            num = self._label_to_num[x]
            label = x
        else:
            assert isinstance(x, int)
            num = x
            label = self._num_to_label[x]
        return num, label

    def edge_existence(self, x, y):
        x_num, x_label = self._get_num_and_label(x)
        y_num, y_label = self._get_num_and_label(y)
        if (x_num, y_num) not in self._sep_sets:
            return "Vertices {} and {} are connected".format(x_label, y_label)
        else:
            p, Z = self._sep_sets[(x_num, y_num)]
            Z = list(map(self._num_to_label.__getitem__, Z))
            return "Vertices {} and {} are d-separated by {} with significance level = {}" \
                .format(x_label, y_label, Z, p)

    def edge_direction(self, x, y):
        x_num, x_label = self._get_num_and_label(x)
        y_num, y_label = self._get_num_and_label(y)
        if (x_num, y_num) in self._sep_sets:
            return "Vertices {} and {} are separated".format(x_label, y_label)
        elif (x_num, y_num) not in self._orientations:
            return 'Edge from {} to {} is not oriented'.format(x_label, y_label)
        else:
            reason = self._orientations[(x_num, y_num)]
            reason = re.sub('\d+', lambda x: repr(self._num_to_label[int(x.group()) - 1]), reason)
            return "Edge from {} to {} is oriented. Reason:\n{}".format(x_label, y_label, reason)

    def get_fetched_log(self, fetched_log_path=None):
        n = len(self._label_to_num)
        with open(fetched_log_path, "w") as fetched_log:
            fetched_log.write("SKELETON\n\n")
            for i in range(n):
                for j in range(i + 1, n):
                    fetched_log.write(self.edge_existence(i, j) + '\n')

            fetched_log.write("\n\nORIENTATION\n\n")
            for i in range(n):
                for j in range(n):
                    if i != j:
                        fetched_log.write(self.edge_direction(i, j) + '\n')
