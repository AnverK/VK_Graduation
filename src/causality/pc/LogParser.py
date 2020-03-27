import os

import parse as parser
import re


def pc_sep_set_type(s):
    if len(s) == 2:
        return []
    return list(map(int, s[1:-1].split(', ')))


def seq_spaced(s):
    return list(map(int, s.split()))


class LogParser:

    def __init__(self, log_path, pvalue, n_long_metrics=4, n_short_metrics=18):
        self.log_path = log_path
        self.pvalue = pvalue
        self.n_long_metrics = n_long_metrics
        self.n_short_metrics = n_short_metrics
        log_file = open(log_path, 'r')
        self.sep_sets = {}
        self.orientations = {}
        self.pc_sep_set_pattern = parser.compile("x = {:d}, y = {:d}, S = {:sep_set_type} : pval = {:g}",
                                                 extra_types=dict(sep_set_type=pc_sep_set_type))
        self.fci_sep_set_pattern = parser.compile("x= {:d}  y= {:d}  S= {:sep_set_type} : pval = {:g}",
                                                  extra_types=dict(sep_set_type=seq_spaced))
        self.rule_pattern = parser.compile("Rule {:d}")
        self.rule_1_pattern = parser.compile("Orient: {:d} *-> {:d} o-* {:seq_type} "
                                             "as: {:d} -> {:seq_type}",
                                             extra_types=dict(seq_type=seq_spaced))
        self.rule_2_pattern = parser.compile("Orient: {:d} -> {:seq_type} *-> {:d} "
                                             "or {:d} *-> {:seq_type} -> {:d} "
                                             "with {:d} *-o {:d} "
                                             "as: {:d} *-> {:d}",
                                             extra_types=dict(seq_type=seq_spaced))

        self.rule_4a_pattern = parser.compile("There is a discriminating path between {:d} and {:d} for {:d} ,"
                                              "and {:d} is not in Sepset of {:d} and {:d} . "
                                              "Orient: {:d} <-> {:d} <-> {:d}",
                                              extra_types=dict(seq_type=seq_spaced))
        self.collider_pattern = parser.compile("{:d} *-> {:d} <-* {:d}")
        self._parse(log_file)
        log_file.close()

    def _parse_pc_skeleton(self, log_file):
        for line in log_file:
            line = line.strip()
            if len(line) == 0:
                break
            if line[0] != 'x':
                continue
            res = self.pc_sep_set_pattern.parse(line)
            p = res[3]
            if p < self.pvalue:
                continue
            x = res[0]
            y = res[1]
            Z = res[2]
            self.sep_sets[(x, y)] = (p, Z)

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
            res = self.fci_sep_set_pattern.parse(line)
            p = res[3]
            if p < self.pvalue:
                continue
            # Alarm! In FCI skeleton log it numerates from 1, not 0.
            x = res[0] - 1
            y = res[1] - 1
            Z = map(lambda x: x - 1, res[2])
            self.sep_sets[(x, y)] = (p, Z)

    # Alarm! In FCI skeleton log it numerates from 1, not 0.
    def _orient_colliders(self, log_file):
        for line in log_file:
            line = line.strip()
            if line.startswith("Rule "):
                return line  # bad design, but ftell() and fseek don't work even in while loop
            res = self.collider_pattern.parse(line)
            if res is not None:
                sepsets = log_file.readline()
                # sepsets = parser.compile('Sxz= {:seq_type} and Szx= {:seq_type}',
                #                          extra_types=dict(seq_type=seq_spaced))
                x = res[0] - 1
                z = res[1] - 1
                y = res[2] - 1
                self.orientations[(x, z)] = 'Collider: ' + line + '\n' + sepsets
                self.orientations[(y, z)] = 'Collider: ' + line + '\n' + sepsets

    # Alarm! In FCI skeleton log it numerates from 1, not 0.
    def _handle_rule(self, n_rule, description):
        reason = 'Rule {}: '.format(n_rule) + description
        if n_rule == 1:
            res = self.rule_1_pattern.parse(description)
            y = res[1] - 1
            Z = list(map(lambda x: x - 1, res[2]))
            for z in Z:
                self.orientations[(y, z)] = reason
                self.orientations[(z, y)] = reason
        if n_rule == 2:
            res = self.rule_2_pattern.parse(description)
            y = res[-2] - 1
            z = res[-1] - 1
            self.orientations[(y, z)] = reason
        if n_rule == 4:
            res = self.rule_4a_pattern.parse(description)
            x = res[-3] - 1
            y = res[-2] - 1
            z = res[-1] - 1
            self.orientations[(x, y)] = reason
            self.orientations[(y, x)] = reason
            self.orientations[(y, z)] = reason
            self.orientations[(z, y)] = reason

    def _orient_with_rules(self, log_file, first_rule=None):
        rule = first_rule
        n_rule = self.rule_pattern.parse(rule)[0]
        description = log_file.readline().strip()
        self._handle_rule(n_rule, description)

        for line in log_file:
            line = line.strip()
            if len(line) == 0:
                continue
            n_rule = self.rule_pattern.parse(line)[0]
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

    def get_number_of_metric(self, x):
        if x < self.n_long_metrics:
            return 'long_{}'.format(x)
        return 'short_{}'.format(x - self.n_long_metrics)

    def edge_existence(self, x, y):
        if (x, y) not in self.sep_sets:
            x = self.get_number_of_metric(x)
            y = self.get_number_of_metric(y)
            return "Vertices {} and {} are connected".format(x, y)
        else:
            p, Z = self.sep_sets[(x, y)]
            x = self.get_number_of_metric(x)
            y = self.get_number_of_metric(y)
            Z = list(map(self.get_number_of_metric, Z))
            return "Vertices {} and {} are d-separated by {} with significance level = {}".format(x, y, Z, p)

    def edge_direction(self, x, y):
        x_name = self.get_number_of_metric(x)
        y_name = self.get_number_of_metric(y)
        if (x, y) in self.sep_sets:
            return "Vertices are separated".format(x_name, y_name)
        elif (x, y) not in self.orientations:
            return 'Edge from {} to {} is not oriented'.format(x_name, y_name)
        else:
            reason = self.orientations[(x, y)]
            reason = re.sub('\d+', lambda x: repr(self.get_number_of_metric(int(x.group()) - 1)), reason)
            return "Edge from {} to {} is oriented. Reason:\n{}".format(x_name, y_name, reason)


n_long = 4
n_short = 18
p = LogParser(os.path.join('pcalg_logs', 'fci_new_p_0.01.log'), 0.01, n_long, n_short)

for i in range(n_long + n_short):
    if i != 0:
        print(p.edge_existence(0, i))
