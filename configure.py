#!/usr/bin/env python3

"""
Configuration program for botan

This script supports Python 3 syntax only. At least CPython 3.10 is recommended.
Latest PyPy3 should also work, but this is only occasionally tested.

(C) 2009-2020 Jack Lloyd
(C) 2015,2016,2017 Simon Warta (Kullo GmbH)
(C) 2019-2022 René Meusel (neXenio GmbH, Rohde & Schwarz Cybersecurity GmbH)

Botan is released under the Simplified BSD License (see license.txt)
"""

import collections
import copy
import json
import sys
import os
import os.path
import platform
import re
import shlex
import shutil
import subprocess
import traceback
import logging
import time
import errno
import optparse # pylint: disable=deprecated-module

# An error caused by and to be fixed by the user, e.g. invalid command line argument
class UserError(Exception):
    pass


# An error caused by bugs in this script or when reading/parsing build data files
# Those are not expected to be fixed by the user of this script
class InternalError(Exception):
    pass


def flatten(l):
    return sum(l, [])

def normalize_source_path(source):
    """
    cmake and some versions of make need this, and nothing else minds
    """
    return os.path.normpath(source).replace('\\', '/')

def normalize_source_paths(sources):
    return [normalize_source_path(p) for p in sources]

def parse_version_file(version_path):
    version_file = open(version_path, encoding='utf8')
    key_and_val = re.compile(r"([a-z_]+) = ([a-zA-Z0-9:\-\']+)")

    results = {}
    for line in version_file.readlines():
        if not line or line[0] == '#':
            continue
        match = key_and_val.match(line)
        if match:
            key = match.group(1)
            val = match.group(2)

            if val == 'None':
                val = None
            elif val.startswith("'") and val.endswith("'"):
                val = val[1:len(val)-1]
            else:
                val = int(val)

            results[key] = val
    return results

class Version:
    """
    Version information are all static members
    """
    data = {}

    @staticmethod
    def get_data():
        if not Version.data:
            root_dir = os.path.dirname(os.path.realpath(__file__))
            Version.data = parse_version_file(os.path.join(root_dir, 'src/build-data/version.txt'))

            suffix = Version.data["release_suffix"]
            if suffix != "":
                suffix_re = re.compile('-(alpha|beta|rc)[0-9]+')

                if not suffix_re.match(suffix):
                    raise Exception("Unexpected version suffix '%s'" % (suffix))
        return Version.data

    @staticmethod
    def major():
        return Version.get_data()["release_major"]

    @staticmethod
    def minor():
        return Version.get_data()["release_minor"]

    @staticmethod
    def patch():
        return Version.get_data()["release_patch"]

    @staticmethod
    def suffix():
        return Version.get_data()["release_suffix"]

    @staticmethod
    def packed():
         # Used on macOS for dylib versioning
        return Version.major() * 1000 + Version.minor()

    @staticmethod
    def so_rev():
        return Version.get_data()["release_so_abi_rev"]

    @staticmethod
    def release_type():
        return Version.get_data()["release_type"]

    @staticmethod
    def datestamp():
        return Version.get_data()["release_datestamp"]

    @staticmethod
    def short_version_string():
        return "%d.%d.%d%s" % (Version.major(), Version.minor(), Version.patch(), Version.suffix())

    @staticmethod
    def full_version_string(options):
        version = "Botan %s" % (Version.short_version_string())

        if options.unsafe_fuzzer_mode or options.unsafe_terminate_on_asserts:
            version += " UNSAFE "
            if options.unsafe_fuzzer_mode:
                version += "FUZZER MODE "
            if options.unsafe_terminate_on_asserts:
                version += "TERMINATE ON ASSERTS "
            version += "BUILD"

        version += " ("
        version += Version.release_type()

        if Version.datestamp() != 0:
            version += ", dated %d" % (Version.datestamp())

        if Version.vc_rev() != "unknown":
            version += ", revision %s" % (Version.vc_rev())

        if options.distribution_info is not None:
            version += ", distribution '%s'" % (options.distribution_info)

        version += ")"

        return version


    @staticmethod
    def as_string():
        return '%d.%d.%d%s' % (Version.major(), Version.minor(), Version.patch(), Version.suffix())

    @staticmethod
    def vc_rev():
        # Lazy load to ensure _local_repo_vc_revision() does not run before logger is set up
        if Version.get_data()["release_vc_rev"] is None:
            Version.data["release_vc_rev"] = Version._local_repo_vc_revision()
        return Version.get_data()["release_vc_rev"]

    @staticmethod
    def _local_repo_vc_revision():
        vc_command = ['git', 'rev-parse', 'HEAD']
        cmdname = vc_command[0]

        try:
            vc = subprocess.Popen(
                vc_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True)
            (stdout, stderr) = vc.communicate()

            if vc.returncode != 0:
                logging.debug('Error getting rev from %s - %d (%s)',
                              cmdname, vc.returncode, stderr)
                return 'unknown'

            rev = str(stdout).strip()
            logging.debug('%s reported revision %s', cmdname, rev)

            return '%s:%s' % (cmdname, rev)
        except OSError as ex:
            logging.debug('Error getting rev from %s - %s', cmdname, ex.strerror)
            return 'unknown'



class SourcePaths:
    """
    A collection of paths defined by the project structure and
    independent of user configurations.
    All paths are relative to the base_dir, which may be relative as well (e.g. ".")
    """

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.doc_dir = os.path.join(self.base_dir, 'doc')
        self.src_dir = os.path.join(self.base_dir, 'src')

        # dirs in src/
        self.build_data_dir = os.path.join(self.src_dir, 'build-data')
        self.configs_dir = os.path.join(self.src_dir, 'configs')
        self.lib_dir = os.path.join(self.src_dir, 'lib')
        self.python_dir = os.path.join(self.src_dir, 'python')
        self.scripts_dir = os.path.join(self.src_dir, 'scripts')

        # subdirs of src/
        self.test_data_dir = os.path.join(self.src_dir, 'tests/data')
        self.sphinx_config_dir = os.path.join(self.configs_dir, 'sphinx')


class BuildPaths:
    """
    Constructor
    """
    def __init__(self, source_paths, options, modules):
        self.build_dir = os.path.join(options.with_build_dir, 'build')

        self.libobj_dir = os.path.join(self.build_dir, 'obj', 'lib')
        self.cliobj_dir = os.path.join(self.build_dir, 'obj', 'cli')
        self.testobj_dir = os.path.join(self.build_dir, 'obj', 'test')

        self.doc_output_dir = os.path.join(self.build_dir, 'docs')
        self.handbook_output_dir = os.path.join(self.doc_output_dir, 'handbook')
        self.doc_output_dir_doxygen = os.path.join(self.doc_output_dir, 'doxygen') if options.with_doxygen else None
        self.doc_module_info = os.path.join(self.build_dir, 'module_info') if options.with_doxygen else None
        self.response_file_dir = os.path.join(self.build_dir, 'response_files')

        # We split the header include paths into 'public', 'internal' and 'external'
        # to allow for better control over what is exposed to each compilation unit.
        # For instance, the examples should only see the public headers, while the
        # test suite should see both public and internal headers.
        self.include_dir = os.path.join(self.build_dir, 'include')
        self.public_include_basedir = os.path.join(self.include_dir, 'public')
        self.internal_include_basedir = os.path.join(self.include_dir, 'internal')
        self.external_include_dir = os.path.join(self.include_dir, 'external')
        self.public_include_dir = os.path.join(self.public_include_basedir, 'botan')
        self.internal_include_dir = os.path.join(self.internal_include_basedir, 'botan', 'internal')

        self.internal_headers = sorted(flatten([m.internal_headers() for m in modules]))
        self.external_headers = sorted(flatten([m.external_headers() for m in modules]))

        # this is overwritten if amalgamation is used
        self.lib_sources = normalize_source_paths(sorted(flatten([mod.sources() for mod in modules])))

        self.public_headers = sorted(flatten([m.public_headers() for m in modules]))

        def find_sources_in(basedir, srcdir):
            for (dirpath, _, filenames) in os.walk(os.path.join(basedir, srcdir)):
                for filename in filenames:
                    if filename.endswith('.cpp') and not filename.startswith('.'):
                        yield os.path.join(dirpath, filename)

        def find_headers_in(basedir, srcdir):
            for (dirpath, _, filenames) in os.walk(os.path.join(basedir, srcdir)):
                for filename in filenames:
                    if filename.endswith('.h') and not filename.startswith('.'):
                        yield os.path.join(dirpath, filename)

        self.cli_sources = normalize_source_paths(find_sources_in(source_paths.src_dir, 'cli'))
        self.cli_headers = normalize_source_paths(find_headers_in(source_paths.src_dir, 'cli'))
        self.test_sources = normalize_source_paths(find_sources_in(source_paths.src_dir, 'tests'))

        if 'examples' in options.build_targets:
            self.example_sources = normalize_source_paths(find_sources_in(source_paths.src_dir, 'examples'))
            self.example_output_dir = os.path.join(self.build_dir, 'examples')
            self.example_obj_dir = os.path.join(self.build_dir, 'obj', 'examples')
        else:
            self.example_sources = None
            self.example_output_dir = None
            self.example_obj_dir = None

        if options.build_fuzzers:
            self.fuzzer_sources = list(find_sources_in(source_paths.src_dir, 'fuzzer'))
            self.fuzzer_output_dir = os.path.join(self.build_dir, 'fuzzer')
            self.fuzzobj_dir = os.path.join(self.build_dir, 'obj', 'fuzzer')
        else:
            self.fuzzer_sources = None
            self.fuzzer_output_dir = None
            self.fuzzobj_dir = None

    def build_dirs(self):
        out = [
            self.libobj_dir,
            self.cliobj_dir,
            self.testobj_dir,
            self.public_include_dir,
            self.internal_include_dir,
            self.external_include_dir,
            self.handbook_output_dir,
            self.response_file_dir
        ]
        if self.doc_output_dir_doxygen:
            out += [self.doc_output_dir_doxygen, self.doc_module_info]
        if self.fuzzer_output_dir:
            out += [self.fuzzobj_dir, self.fuzzer_output_dir]
        if self.example_output_dir:
            out += [self.example_obj_dir, self.example_output_dir]
        return out

    def format_public_include_flags(self, cc):
        return cc.add_include_dir_option + ' ' + normalize_source_path(self.public_include_basedir)

    def format_internal_include_flags(self, cc):
        return cc.add_include_dir_option + ' ' + normalize_source_path(self.internal_include_basedir)

    def format_external_include_flags(self, cc, external_includes):
        dash_isystem = cc.add_system_include_dir_option
        output = ''
        if self.external_headers:
            output += ' ' + dash_isystem + ' ' + normalize_source_path(self.external_include_dir)
        for external_include in external_includes:
            output += ' ' + dash_isystem + ' ' + normalize_source_path(external_include)
        return output

    def src_info(self, typ):
        if typ == 'lib':
            return (self.lib_sources, self.libobj_dir)
        if typ == 'cli':
            return (self.cli_sources, self.cliobj_dir)
        if typ == 'test':
            return (self.test_sources, self.testobj_dir)
        if typ == 'fuzzer':
            return (self.fuzzer_sources, self.fuzzobj_dir)
        if typ == 'examples':
            return (self.example_sources, self.example_obj_dir)
        raise InternalError("Unknown src info type '%s'" % (typ))

ACCEPTABLE_BUILD_TARGETS = ["static", "shared", "cli", "tests", "bogo_shim", "examples", "ct_selftest"]

def process_command_line(args):
    """
    Handle command line options
    Do not use logging in this method as command line options need to be
    available before logging is setup.
    """

    def define_option_pair(group, verb, nverb, what, default, msg=optparse.SUPPRESS_HELP):
        dest = '%s_%s' % (verb, what.replace('-', '_'))

        # always show the help for the option that switches away from the default
        with_help = msg if not default else optparse.SUPPRESS_HELP
        without_help = msg if default else optparse.SUPPRESS_HELP

        group.add_option('--%s-%s' % (verb, what),
                         dest=dest,
                         action='store_true',
                         default=default,
                         help=with_help)

        group.add_option('--%s-%s' % (nverb, what),
                         dest=dest,
                         action='store_false',
                         help=without_help)

    def add_with_without_pair(group, what, default, msg=optparse.SUPPRESS_HELP):
        define_option_pair(group, 'with', 'without', what, default, msg)

    def add_enable_disable_pair(group, what, default, msg=optparse.SUPPRESS_HELP):
        define_option_pair(group, 'enable', 'disable', what, default, msg)

    parser = optparse.OptionParser(
        formatter=optparse.IndentedHelpFormatter(max_help_position=50),
        version=Version.as_string())

    parser.add_option('--verbose', action='store_true', default=False,
                      help='Show debug messages')
    parser.add_option('--quiet', action='store_true', default=False,
                      help='Show only warnings and errors')

    target_group = optparse.OptionGroup(parser, 'Target options')

    target_group.add_option('--cpu', help='set the target CPU architecture')

    target_group.add_option('--os', help='set the target operating system')

    target_group.add_option('--cc', dest='compiler', help='set the desired build compiler')

    target_group.add_option('--cc-min-version', dest='cc_min_version', default=None,
                            metavar='MAJOR.MINOR',
                            help='Set the minimal version of the target compiler. ' \
                                 'Use --cc-min-version=0.0 to support all compiler versions. ' \
                                 'Default is auto detection.')

    target_group.add_option('--cc-bin', dest='compiler_binary', metavar='BINARY',
                            help='set path to compiler binary')

    target_group.add_option('--cc-abi-flags', metavar='FLAGS', default='',
                            help='set compiler ABI flags')

    target_group.add_option('--cxxflags', metavar='FLAGS', default=None,
                            help='override all compiler flags')

    target_group.add_option('--extra-cxxflags', metavar='FLAGS', default=[], action='append',
                            help='set extra compiler flags')

    target_group.add_option('--lto-cxxflags-to-ldflags', default=False, action='store_true',
                            help='set all compilation flags also during linking (for LTO)')

    target_group.add_option('--ldflags', metavar='FLAGS',
                            help='set linker flags', default=None)

    target_group.add_option('--extra-libs', metavar='LIBS',
                            help='specify extra libraries to link against', default='')

    target_group.add_option('--ar-command', dest='ar_command', metavar='AR', default=None,
                            help='set path to static archive creator')

    target_group.add_option('--ar-options', dest='ar_options', metavar='AR_OPTIONS', default=None,
                            help='set options for ar')

    target_group.add_option('--msvc-runtime', metavar='RT', default=None,
                            help='specify MSVC runtime (MT, MD, MTd, MDd)')

    target_group.add_option('--compiler-cache',
                            help='specify a compiler cache to use')

    target_group.add_option('--ct-value-barrier-type', metavar='TYPE', default=None,
                            help=optparse.SUPPRESS_HELP)

    target_group.add_option('--with-os-features', action='append', metavar='FEAT',
                            help='specify OS features to use')
    target_group.add_option('--without-os-features', action='append', metavar='FEAT',
                            help='specify OS features to disable')

    add_with_without_pair(target_group, 'compilation-database', True, 'disable compile_commands.json')

    build_group = optparse.OptionGroup(parser, 'Build options')

    build_group.add_option('--system-cert-bundle', metavar='PATH', default=None,
                           help='set path to trusted CA bundle')

    add_with_without_pair(build_group, 'debug-info', False, 'include debug symbols')

    add_with_without_pair(build_group, 'sanitizers', False, 'enable ASan/UBSan checks')

    add_enable_disable_pair(build_group, 'asm', True, 'disable inline assembly')

    add_enable_disable_pair(build_group, 'stack-scrubbing', False, 'enable compiler-assisted stack scrubbing')

    build_group.add_option('--enable-sanitizers', metavar='SAN', default='',
                           help='enable specific sanitizers')

    add_with_without_pair(build_group, 'stack-protector', None, 'disable stack smashing protections')

    add_with_without_pair(build_group, 'coverage-info', False, 'add coverage info')

    build_group.add_option('--enable-shared-library', dest='build_shared_lib',
                           action='store_true', default=None,
                           help=optparse.SUPPRESS_HELP)
    build_group.add_option('--disable-shared-library', dest='build_shared_lib',
                           action='store_false',
                           help='disable building shared library')

    build_group.add_option('--enable-static-library', dest='build_static_lib',
                           action='store_true', default=None,
                           help=optparse.SUPPRESS_HELP)
    build_group.add_option('--disable-static-library', dest='build_static_lib',
                           action='store_false',
                           help='disable building static library')

    build_group.add_option('--optimize-for-size', dest='optimize_for_size',
                           action='store_true', default=False,
                           help='optimize for code size')

    build_group.add_option('--no-optimizations', dest='no_optimizations',
                           action='store_true', default=False,
                           help='disable all optimizations (for debugging)')

    build_group.add_option('--debug-mode', action='store_true', default=False, dest='debug_mode',
                           help='enable debug info, disable optimizations')

    build_group.add_option('--amalgamation', dest='amalgamation',
                           default=False, action='store_true',
                           help='use amalgamation to build')

    build_group.add_option('--name-amalgamation', metavar='NAME', default='botan_all',
                           help='specify alternate name for amalgamation files')

    build_group.add_option('--with-build-dir', metavar='DIR', default='',
                           help='setup the build in DIR')

    build_group.add_option('--with-external-includedir', metavar='DIR', default=[],
                           help='use DIR for external includes', action='append')

    build_group.add_option('--with-external-libdir', metavar='DIR', default=[],
                           help='use DIR for external libs', action='append')

    build_group.add_option('--define-build-macro', metavar='DEFINE', default=[],
                           help='set compile-time pre-processor definition like KEY[=VALUE]', action='append')

    build_group.add_option('--with-sysroot-dir', metavar='DIR', default='',
                           help='use DIR for system root while cross-compiling')

    link_methods = ['symlink', 'hardlink', 'copy']
    build_group.add_option('--link-method', default=None, metavar='METHOD',
                           choices=link_methods,
                           help='choose how links to include headers are created (%s)' % ', '.join(link_methods))

    build_group.add_option('--distribution-info', metavar='STRING',
                           help='distribution specific version', default=None)

    build_group.add_option('--maintainer-mode', dest='maintainer_mode',
                           action='store_true', default=False,
                           help=optparse.SUPPRESS_HELP)

    build_group.add_option('--werror-mode', dest='werror_mode',
                           action='store_true', default=False,
                           help="Prohibit compiler warnings")

    build_group.add_option('--no-install-python-module', action='store_true', default=False,
                           help='skip installing Python module')

    build_group.add_option('--with-python-versions', dest='python_version',
                           metavar='N.M',
                           default='%d.%d' % (sys.version_info[0], sys.version_info[1]),
                           help='where to install botan2.py (def %default)')

    build_group.add_option('--disable-cc-tests', dest='enable_cc_tests',
                           default=True, action='store_false',
                           help=optparse.SUPPRESS_HELP)

    add_with_without_pair(build_group, 'valgrind', False, 'use valgrind API')

    build_group.add_option('--unsafe-fuzzer-mode', action='store_true', default=False,
                           help=optparse.SUPPRESS_HELP)

    build_group.add_option('--build-fuzzers', dest='build_fuzzers',
                           metavar='TYPE', default=None,
                           help='Build fuzzers (afl, libfuzzer, klee, test)')

    build_group.add_option('--with-fuzzer-lib', metavar='LIB', default=None, dest='fuzzer_lib',
                           help='additionally link in LIB')

    add_with_without_pair(build_group, 'debug-asserts', default=False)

    build_group.add_option('--unsafe-terminate-on-asserts', action='store_true', default=False,
                           help=optparse.SUPPRESS_HELP)

    build_group.add_option('--build-targets', default=None, dest="build_targets", action='append',
                           help="build specific targets and tools (%s)" % ', '.join(ACCEPTABLE_BUILD_TARGETS))

    build_group.add_option('--build-tool', default='make',
                           help="specify the build tool (make, ninja)")

    add_with_without_pair(build_group, 'pkg-config', default=None)
    add_with_without_pair(build_group, 'cmake-config', default=True)

    docs_group = optparse.OptionGroup(parser, 'Documentation Options')

    add_with_without_pair(docs_group, 'documentation', True, 'skip building/installing documentation')

    add_with_without_pair(docs_group, 'sphinx', None, 'run Sphinx to generate docs')

    add_with_without_pair(docs_group, 'pdf', False, 'run Sphinx to generate PDF doc')

    add_with_without_pair(docs_group, 'rst2man', None, 'run rst2man to generate man page')

    add_with_without_pair(docs_group, 'doxygen', False, 'run Doxygen')

    mods_group = optparse.OptionGroup(parser, 'Module selection')

    mods_group.add_option('--module-policy', dest='module_policy',
                          help="module policy file (see src/build-data/policy)",
                          metavar='POL', default=None)

    mods_group.add_option('--enable-modules', dest='enabled_modules',
                          metavar='MODS', action='append',
                          help='enable specific modules')
    mods_group.add_option('--disable-modules', dest='disabled_modules',
                          metavar='MODS', action='append',
                          help='disable specific modules')
    mods_group.add_option('--no-autoload', action='store_true', default=False,
                          help=optparse.SUPPRESS_HELP)
    mods_group.add_option('--minimized-build', action='store_true', dest='no_autoload',
                          help='minimize build')

    add_enable_disable_pair(mods_group, 'experimental-features', False,
                            'enable building of experimental features and modules')

    add_enable_disable_pair(mods_group, 'deprecated-features', True,
                            'disable building of deprecated features and modules')

    # Should be derived from info.txt but this runs too early
    third_party = ['boost', 'bzip2', 'esdm_rng', 'lzma', 'commoncrypto', 'sqlite3', 'zlib', 'tpm', 'tpm2']

    for mod in third_party:
        mods_group.add_option('--with-%s' % (mod),
                              help=('use %s' % (mod)) if mod in third_party else optparse.SUPPRESS_HELP,
                              action='append_const',
                              const=mod,
                              dest='enabled_modules')

        mods_group.add_option('--without-%s' % (mod),
                              help=optparse.SUPPRESS_HELP,
                              action='append_const',
                              const=mod,
                              dest='disabled_modules')

    install_group = optparse.OptionGroup(parser, 'Installation options')

    install_group.add_option('--program-suffix', metavar='SUFFIX',
                             help='append string to program names')
    install_group.add_option('--library-suffix', metavar='SUFFIX', default='',
                             help='append string to library names')

    install_group.add_option('--prefix', metavar='DIR',
                             help='set the install prefix')
    install_group.add_option('--docdir', metavar='DIR',
                             help='set the doc install dir')
    install_group.add_option('--bindir', metavar='DIR',
                             help='set the binary install dir')
    install_group.add_option('--libdir', metavar='DIR',
                             help='set the library install dir')
    install_group.add_option('--mandir', metavar='DIR',
                             help='set the install dir for man pages')
    install_group.add_option('--includedir', metavar='DIR',
                             help='set the include file install dir')

    info_group = optparse.OptionGroup(parser, 'Informational')

    info_group.add_option('--list-modules', dest='list_modules',
                          action='store_true',
                          help='list available modules and exit')

    info_group.add_option('--list-os-features', dest='list_os_features',
                          action='store_true',
                          help='list available OS features and exit')

    parser.add_option_group(target_group)
    parser.add_option_group(build_group)
    parser.add_option_group(docs_group)
    parser.add_option_group(mods_group)
    parser.add_option_group(install_group)
    parser.add_option_group(info_group)

    # These exist only for autoconf compatibility (requested by zw for mtn)
    compat_with_autoconf_options = [
        'datadir',
        'datarootdir',
        'dvidir',
        'exec-prefix',
        'htmldir',
        'infodir',
        'libexecdir',
        'localedir',
        'localstatedir',
        'oldincludedir',
        'pdfdir',
        'psdir',
        'sbindir',
        'sharedstatedir',
        'sysconfdir'
        ]

    for opt in compat_with_autoconf_options:
        parser.add_option('--' + opt, help=optparse.SUPPRESS_HELP)

    (options, args) = parser.parse_args(args)

    if args != []:
        raise UserError('Unhandled option(s): ' + ' '.join(args))

    if options.debug_mode:
        options.no_optimizations = True
        options.with_debug_info = True

    def parse_multiple_enable(modules):
        if modules is None:
            return []

        return sorted({m for m in flatten([s.split(',') for s in modules]) if m != ''})

    options.enabled_modules = parse_multiple_enable(options.enabled_modules)
    options.disabled_modules = parse_multiple_enable(options.disabled_modules)

    options.with_os_features = parse_multiple_enable(options.with_os_features)
    options.without_os_features = parse_multiple_enable(options.without_os_features)

    return options

def take_options_from_env(options):
    # Take some values from environment, if not set on command line

    def update_from_env(val, var, name):
        if val is None:
            val = os.getenv(var)
            if val is not None:
                logging.info('Implicit --%s=%s due to environment variable %s', name, val, var)

        return val

    if os.getenv('CXX') and options.compiler_binary is None and options.compiler is not None:
        logging.info('CXX environment variable is set which will override compiler path')

    options.ar_command = update_from_env(options.ar_command, 'AR', 'ar-command')
    options.ar_options = update_from_env(options.ar_options, 'AR_OPTIONS', 'ar-options')
    options.compiler_binary = update_from_env(options.compiler_binary, 'CXX', 'cc-bin')
    options.cxxflags = update_from_env(options.cxxflags, 'CXXFLAGS', 'cxxflags')
    options.ldflags = update_from_env(options.ldflags, 'LDFLAGS', 'ldflags')

class LexResult:
    pass


class LexerError(InternalError):
    def __init__(self, msg, lexfile, line):
        super().__init__(msg)
        self.msg = msg
        self.lexfile = lexfile
        self.line = line

    def __str__(self):
        return '%s at %s:%d' % (self.msg, self.lexfile, self.line)

def parse_lex_dict(as_list, map_name, infofile):
    if len(as_list) % 3 != 0:
        raise InternalError("Lex dictionary has invalid format (input not divisible by 3): %s" % as_list)

    result = {}
    for key, sep, value in [as_list[3*i:3*i+3] for i in range(0, len(as_list)//3)]:
        if sep != '->':
            raise InternalError("Map %s in %s has invalid format" % (map_name, infofile))
        if key in result:
            raise InternalError("Duplicate map entry %s in map %s file %s" % (key, map_name, infofile))
        result[key] = value
    return result

def lex_me_harder(infofile, allowed_groups, allowed_maps, name_val_pairs):
    """
    Generic lexer function for info.txt and src/build-data files
    """
    out = LexResult()

    # Format as a nameable Python variable
    def py_var(group):
        return group.replace(':', '_')

    lexer = shlex.shlex(open(infofile, encoding='utf8'), infofile, posix=True)
    lexer.wordchars += '=:.<>/,-!?+*' # handle various funky chars in info.txt

    groups = allowed_groups + allowed_maps
    for group in groups:
        out.__dict__[py_var(group)] = []
    for (key, val) in name_val_pairs.items():
        out.__dict__[key] = val

    def lexed_tokens(): # Convert to an iterator
        while True:
            token = lexer.get_token()
            if token != lexer.eof:
                yield token
            else:
                return

    for token in lexed_tokens():
        match = re.match('<(.*)>', token)

        # Check for a grouping
        if match is not None:
            group = match.group(1)

            if group not in groups:
                raise LexerError('Unknown group "%s"' % (group),
                                 infofile, lexer.lineno)

            end_marker = '</' + group + '>'

            token = lexer.get_token()
            while token != end_marker:
                out.__dict__[py_var(group)].append(token)
                token = lexer.get_token()
                if token is None:
                    raise LexerError('Group "%s" not terminated' % (group),
                                     infofile, lexer.lineno)

        elif token in name_val_pairs.keys():
            if isinstance(out.__dict__[token], list):
                out.__dict__[token].append(lexer.get_token())
            else:
                out.__dict__[token] = lexer.get_token()

        else: # No match -> error
            raise LexerError('Bad token "%s"' % (token), infofile, lexer.lineno)

    for group in allowed_maps:
        out.__dict__[group] = parse_lex_dict(out.__dict__[group], group, infofile)

    return out

class InfoObject:
    def __init__(self, infofile):
        """
        Constructor sets members `infofile`, `lives_in`, `parent_module` and `basename`
        """

        self.infofile = infofile
        (dirname, basename) = os.path.split(infofile)
        self.lives_in = dirname
        if basename == 'info.txt':
            (next_dir, self.basename) = os.path.split(dirname)
            self.parent_module = None

            obj_dir = ''
            while next_dir != obj_dir:
                obj_dir = next_dir
                if os.access(os.path.join(obj_dir, 'info.txt'), os.R_OK):
                    self.parent_module = os.path.basename(obj_dir)
                    break
                (next_dir, _) = os.path.split(obj_dir)
                if os.path.basename(next_dir) == 'src':
                    break
        else:
            self.basename = basename.replace('.txt', '')


class ModuleInfo(InfoObject):
    """
    Represents the information about a particular module
    """

    def __init__(self, infofile):
        super().__init__(infofile)
        lex = lex_me_harder(
            infofile,
            ['header:internal', 'header:public', 'header:external', 'requires',
             'os_features', 'arch', 'isa', 'cc', 'comment', 'warning'],
            ['defines', 'internal_defines', 'libs', 'frameworks', 'module_info'],
            {
                'load_on': 'auto',
            })

        def check_header_duplicates(header_list_public, header_list_internal):
            pub_header = set(header_list_public)
            int_header = set(header_list_internal)
            if not pub_header.isdisjoint(int_header):
                logging.error("Module %s has same header in public and internal sections",
                              self.infofile)

        check_header_duplicates(lex.header_public, lex.header_internal)

        all_source_files = []
        all_header_files = []

        for fspath in os.listdir(self.lives_in):
            if fspath.endswith('.cpp'):
                all_source_files.append(fspath)
            elif fspath.endswith('.h'):
                all_header_files.append(fspath)

        self.source = all_source_files

        # If not entry for the headers, all are assumed internal
        if lex.header_internal == [] and lex.header_public == []:
            self.header_public = []
            self.header_internal = list(all_header_files)
        else:
            self.header_public = lex.header_public
            self.header_internal = lex.header_internal
        self.header_external = lex.header_external

        def convert_lib_list(libs):
            out = {}
            for (os_name, lib_list) in libs.items():
                out[os_name] = lib_list.split(',')
            return out

        def combine_lines(c):
            return ' '.join(c) if c else None

        # Convert remaining lex result to members
        self.arch = lex.arch
        self.cc = lex.cc
        self.comment = combine_lines(lex.comment)
        self._defines = lex.defines
        self._validate_defines_content(self._defines)
        self._internal_defines = lex.internal_defines
        self._validate_defines_content(self._internal_defines)
        self.frameworks = convert_lib_list(lex.frameworks)
        self.libs = convert_lib_list(lex.libs)
        self.load_on = lex.load_on
        self.isa = lex.isa
        self.os_features = lex.os_features
        self.requires = lex.requires
        self.warning = combine_lines(lex.warning)
        self._parse_module_info(lex)

        # Modify members
        self.source = [normalize_source_path(os.path.join(self.lives_in, s)) for s in self.source]
        self.header_internal = [os.path.join(self.lives_in, s) for s in self.header_internal]
        self.header_public = [os.path.join(self.lives_in, s) for s in self.header_public]
        self.header_external = [os.path.join(self.lives_in, s) for s in self.header_external]

        # Filesystem read access check
        for src in self.source + self.header_internal + self.header_public + self.header_external:
            if not os.access(src, os.R_OK):
                logging.error("Missing file %s in %s", src, infofile)

        # Check for duplicates
        def intersect_check(type_a, list_a, type_b, list_b):
            intersection = set.intersection(set(list_a), set(list_b))
            if intersection:
                logging.error('Headers %s marked both %s and %s', ' '.join(intersection), type_a, type_b)

        intersect_check('public', self.header_public, 'internal', self.header_internal)
        intersect_check('public', self.header_public, 'external', self.header_external)
        intersect_check('external', self.header_external, 'internal', self.header_internal)

        # Check module type constraints
        source_file_count = len(all_source_files) + len(all_header_files)
        if self.is_virtual() and source_file_count > 0:
            logging.error("Module '%s' is virtual but contains %d source code files", self.basename, source_file_count)

    def _parse_module_info(self, lex):
        info = lex.module_info

        if "name" not in info:
            raise InternalError("Module '%s' does not contain a <module_info> section with at least a documentation-friendly 'name' definition" % self.basename)

        self.name = info["name"]
        self.brief = info.get("brief") # possibly None
        self.type = info.get("type") or "Public"
        self.lifecycle = info.get("lifecycle") or "Stable"

        if self.type not in ["Public", "Internal", "Virtual"]:
            raise InternalError("Module '%s' has an unknown type: %s" % (self.basename, self.type))
        if self.lifecycle not in ["Stable", "Experimental", "Deprecated"]:
            raise InternalError("Module '%s' has an unknown lifecycle status: %s" % (self.basename, self.lifecycle))

    @staticmethod
    def _validate_defines_content(defines):
        for key, value in defines.items():
            if not re.match('^[0-9A-Za-z_]{3,30}$', key):
                raise InternalError('Module defines key has invalid format: "%s"' % key)
            if not re.match('^20[0-9]{6}$', value):
                raise InternalError('Module defines value has invalid format: "%s" (should be YYYYMMDD)' % value)

            year = int(value[0:4])
            month = int(value[4:6])
            day = int(value[6:])

            if year < 2013 or month == 0 or month > 12 or day == 0 or day > 31:
                raise InternalError('Module defines value has invalid format: "%s" (should be YYYYMMDD)' % value)

    def cross_check(self, arch_info, cc_info, all_os_features, all_isa_extn):

        for feat in set(flatten([o.split(',') for o in self.os_features])):
            if feat not in all_os_features:
                logging.error("Module %s uses an OS feature (%s) which no OS supports", self.infofile, feat)

        for supp_cc in self.cc:
            if supp_cc not in cc_info:
                if supp_cc.startswith('!') and supp_cc[1:] in cc_info:
                    continue

                colon_idx = supp_cc.find(':')
                # a versioned compiler dependency
                if colon_idx > 0 and supp_cc[0:colon_idx] in cc_info:
                    pass
                else:
                    raise InternalError('Module %s mentions unknown compiler %s' % (self.infofile, supp_cc))

        for supp_arch in self.arch:
            if supp_arch not in arch_info:
                raise InternalError('Module %s mentions unknown arch %s' % (self.infofile, supp_arch))

        def known_isa(isa):
            if isa in all_isa_extn:
                return True

            compound_isa = isa.split(':')
            if len(compound_isa) == 2 and compound_isa[0] in arch_info and compound_isa[1] in all_isa_extn:
                return True
            return False

        for isa in self.isa:
            if not known_isa(isa):
                raise InternalError('Module %s uses unknown ISA extension %s' % (self.infofile, isa))

    def sources(self):
        return self.source

    def public_headers(self):
        return self.header_public

    def internal_headers(self):
        return self.header_internal

    def external_headers(self):
        return self.header_external

    def isas_needed(self, arch):
        isas = []

        for isa in self.isa:
            if isa.find(':') == -1:
                isas.append(isa)
            elif isa.startswith(arch + ':'):
                isas.append(isa[len(arch)+1:])

        return isas

    def defines(self):
        return [(key + ' ' + value) for key, value in self._defines.items()]

    def internal_defines(self):
        return [(key + ' ' + value) for key, value in self._internal_defines.items()]

    def compatible_cpu(self, archinfo, options):
        arch_name = archinfo.basename
        cpu_name = options.arch

        for isa in self.isa:
            if isa.find(':') > 0:
                (arch, isa) = isa.split(':')

                if arch != arch_name:
                    continue

            if isa not in archinfo.isa_extensions:
                return False

        if self.arch != []:
            if arch_name not in self.arch and cpu_name not in self.arch:
                return False

        return True

    def compatible_os(self, os_data, options):
        if not self.os_features:
            return True

        def has_all(needed, provided):
            for n in needed:
                if n not in provided:
                    return False
            return True

        provided_features = os_data.enabled_features(options)

        for feature_set in self.os_features:
            if has_all(feature_set.split(','), provided_features):
                return True

        return False

    def compatible_compiler(self, ccinfo, cc_min_version, arch):
        # Check if this compiler supports the flags we need
        def supported_isa_flags(ccinfo, arch):
            for isa in self.isa:
                if ccinfo.isa_flags_for(isa, arch) is None:
                    return False
            return True

        # Check if module gives explicit compiler dependencies
        def supported_compiler(ccinfo, cc_min_version):
            if self.cc == []:
                # no compiler restriction
                return True

            if ccinfo.basename in self.cc:
                # compiler is supported, independent of version
                return True

            if '!%s' % (ccinfo.basename) in self.cc:
                # an explicit exclusion of this compiler
                return False

            # If just exclusions are given, treat as accept if we do not match
            if all(cc.startswith('!') for cc in self.cc):
                return True

            # Maybe a versioned compiler dep
            for cc in self.cc:
                try:
                    name, version = cc.split(":")
                    if name == ccinfo.basename:
                        min_cc_version = [int(v) for v in version.split('.')]
                        cur_cc_version = [int(v) for v in cc_min_version.split('.')]
                        # With lists of ints, this does what we want
                        return cur_cc_version >= min_cc_version
                except ValueError:
                    # No version part specified
                    pass

            return False # compiler not listed

        return supported_isa_flags(ccinfo, arch) and supported_compiler(ccinfo, cc_min_version)

    def dependencies(self, osinfo, archinfo):
        # base is an implicit dep for all submodules
        deps = ['base']
        if self.parent_module is not None:
            deps.append(self.parent_module)

        for req in self.requires:
            if req.find('?') != -1:
                (cond, dep) = req.split('?')
                if osinfo is None and archinfo is None:
                    deps.append(dep)
                else:
                    if cond == archinfo.basename or cond in osinfo.target_features:
                        deps.append(dep)
            else:
                deps.append(req)

        return deps

    def dependencies_exist(self, modules):
        """
        Ensure that all dependencies of this module actually exist, warning
        about any that do not
        """

        def is_dependency_on_virtual(this_module, dependency):
            if not dependency.is_virtual():
                return False

            if this_module.parent_module == dependency.basename:
                return False

            return True

        missing = [s for s in self.dependencies(None, None) if s not in modules or is_dependency_on_virtual(self, modules[s])]

        for modname in missing:
            if modname not in modules:
                logging.error("Module '%s', dep of '%s', does not exist",
                              missing, self.basename)
            else:
                assert modules[modname].is_virtual()
                logging.error("Module '%s' is virtual and cannot be depended on by '%s'",
                              modname, self.basename)

    def is_public(self):
        return self.type == "Public"

    def is_internal(self):
        return self.type == "Internal"

    def is_virtual(self):
        return self.type == "Virtual"

    def is_stable(self):
        return self.lifecycle == "Stable"

    def is_experimental(self):
        return self.lifecycle == "Experimental"

    def is_deprecated(self):
        return self.lifecycle == "Deprecated"

class ModulePolicyInfo(InfoObject):
    def __init__(self, infofile):
        super().__init__(infofile)
        lex = lex_me_harder(
            infofile,
            ['required', 'if_available', 'prohibited'],
            [],
            {})

        self.if_available = lex.if_available
        self.required = lex.required
        self.prohibited = lex.prohibited

    def cross_check(self, modules):
        def check(tp, lst, required):
            msg = "Module policy %s includes non-existent module %s in <%s>"

            for mod in lst:
                if mod not in modules:
                    if required:
                        logging.error(msg, self.infofile, mod, tp)
                    else:
                        logging.warning(msg, self.infofile, mod, tp)

        check('required', self.required, True)
        check('if_available', self.if_available, False)
        check('prohibited', self.prohibited, False)


class ArchInfo(InfoObject):
    def __init__(self, infofile):
        super().__init__(infofile)
        lex = lex_me_harder(
            infofile,
            ['aliases', 'isa_extensions'],
            [],
            {
                'family': None,
            })

        self.aliases = lex.aliases
        self.family = lex.family
        self.isa_extensions = lex.isa_extensions

        alphanumeric = re.compile('^[a-z0-9]+$')
        for isa in self.isa_extensions:
            if alphanumeric.match(isa) is None:
                logging.error('Invalid name for ISA extension "%s"', isa)

    def supported_isa_extensions(self, cc):
        isas = []

        for isa in self.isa_extensions:
            if cc.isa_flags_for(isa, self.basename) is not None:
                isas.append(isa)

        return sorted(isas)


class CompilerInfo(InfoObject):
    def __init__(self, infofile):
        super().__init__(infofile)
        lex = lex_me_harder(
            infofile,
            [],
            ['cpu_flags', 'cpu_flags_no_debug', 'so_link_commands', 'binary_link_commands',
             'mach_abi_linking', 'isa_flags', 'sanitizers', 'lib_flags', 'ct_value_barrier'],
            {
                'binary_name': None,
                'linker_name': None,
                'macro_name': None,
                'minimum_supported_version': None,
                'output_to_object': '-o ',
                'output_to_exe': '-o ',
                'add_include_dir_option': '-I',
                'add_system_include_dir_option': '-I',
                'add_lib_dir_option': '-L',
                'add_compile_definition_option': '-D',
                'add_sysroot_option': '',
                'add_lib_option': '-l%s',
                'add_framework_option': '-framework ',
                'preproc_flags': '-E',
                'compile_flags': '-c',
                'debug_info_flags': '-g',
                'optimization_flags': '',
                'size_optimization_flags': '',
                'sanitizer_optimization_flags': '',
                'coverage_flags': '',
                'stack_protector_flags': '',
                'shared_flags': '',
                'lang_flags': '',
                'lang_binary_linker_flags': '',
                'warning_flags': '',
                'maintainer_warning_flags': '',
                'visibility_build_flags': '',
                'visibility_attribute': '',
                'ar_command': '',
                'ar_options': '',
                'ar_output_to': '',
                'werror_flags': '',
                'supports_gcc_inline_asm': 'no',
                'ninja_header_deps_style': '',
                'header_deps_flag': '',
                'header_deps_out': '',
            })

        self.add_framework_option = lex.add_framework_option
        self.add_include_dir_option = lex.add_include_dir_option
        self.add_system_include_dir_option = lex.add_system_include_dir_option
        self.add_lib_dir_option = lex.add_lib_dir_option
        self.add_lib_option = lex.add_lib_option
        self.add_compile_definition_option = lex.add_compile_definition_option
        self.add_sysroot_option = lex.add_sysroot_option
        self.ar_command = lex.ar_command
        self.ar_options = lex.ar_options
        self.ar_output_to = lex.ar_output_to
        self.binary_link_commands = lex.binary_link_commands
        self.binary_name = lex.binary_name
        self.cpu_flags = lex.cpu_flags
        self.cpu_flags_no_debug = lex.cpu_flags_no_debug
        self.compile_flags = lex.compile_flags
        self.coverage_flags = lex.coverage_flags
        self.debug_info_flags = lex.debug_info_flags
        self.isa_flags = lex.isa_flags
        self.lang_flags = lex.lang_flags
        self.lang_binary_linker_flags = lex.lang_binary_linker_flags
        self.lib_flags = lex.lib_flags
        self.linker_name = lex.linker_name
        self.mach_abi_linking = lex.mach_abi_linking
        self.macro_name = lex.macro_name
        self.maintainer_warning_flags = lex.maintainer_warning_flags
        self.optimization_flags = lex.optimization_flags
        self.output_to_exe = lex.output_to_exe
        self.output_to_object = lex.output_to_object
        self.preproc_flags = lex.preproc_flags
        self.sanitizers = lex.sanitizers
        self.sanitizer_types = []
        self.sanitizer_optimization_flags = lex.sanitizer_optimization_flags
        self.shared_flags = lex.shared_flags
        self.size_optimization_flags = lex.size_optimization_flags
        self.so_link_commands = lex.so_link_commands
        self.stack_protector_flags = lex.stack_protector_flags
        self.supports_gcc_inline_asm = lex.supports_gcc_inline_asm == 'yes'
        self.visibility_attribute = lex.visibility_attribute
        self.visibility_build_flags = lex.visibility_build_flags
        self.warning_flags = lex.warning_flags
        self.werror_flags = lex.werror_flags
        self.minimum_supported_version = lex.minimum_supported_version
        self.ninja_header_deps_style = lex.ninja_header_deps_style
        self.header_deps_flag = lex.header_deps_flag
        self.header_deps_out = lex.header_deps_out
        self.ct_value_barrier = lex.ct_value_barrier

    def cross_check(self, os_info, arch_info, all_isas):

        for isa in self.isa_flags:
            if ":" in isa:
                (arch, isa) = isa.split(":")
                if isa not in all_isas:
                    raise InternalError('Compiler %s has flags for unknown ISA %s' % (self.infofile, isa))
                if arch not in arch_info:
                    raise InternalError('Compiler %s has flags for unknown arch/ISA %s:%s' % (self.infofile, arch, isa))

        for os_name in self.binary_link_commands:
            if os_name in ["default", "default-debug"]:
                continue
            if os_name not in os_info:
                raise InternalError("Compiler %s has binary_link_command for unknown OS %s" % (self.infofile, os_name))

        for os_name in self.so_link_commands:
            if os_name in ["default", "default-debug"]:
                continue
            if os_name not in os_info:
                raise InternalError("Compiler %s has so_link_command for unknown OS %s" % (self.infofile, os_name))

    def isa_flags_for(self, isa, arch):
        if isa.find(':') > 0:
            (isa_arch, isa) = isa.split(':')
            if isa_arch != arch:
                return ''
            if isa in self.isa_flags:
                return self.isa_flags[isa]

        if isa in self.isa_flags:
            return self.isa_flags[isa]
        arch_isa = '%s:%s' % (arch, isa)
        if arch_isa in self.isa_flags:
            return self.isa_flags[arch_isa]

        return None

    def get_isa_specific_flags(self, isas, arch):
        flags = set()

        def simd32_impl():
            for simd_isa in ['ssse3', 'altivec', 'neon']:
                if simd_isa in arch.isa_extensions and \
                   self.isa_flags_for(simd_isa, arch.basename):
                    return simd_isa
            return None

        for isa in isas:

            if isa == 'simd':
                isa = simd32_impl()

                if isa is None:
                    continue

            flagset = self.isa_flags_for(isa, arch.basename)
            if flagset is None:
                raise UserError('Compiler %s does not support %s' % (self.basename, isa))
            flags.add(flagset)

        return " ".join(sorted(flags))

    def gen_lib_flags(self, options, variables):
        """
        Return any flags specific to building the library
        (vs the cli or tests)
        """

        def flag_builder():
            # We always emit -fPIC or equivalent so that position independent executables
            # can be created that link to the static library
            yield self.shared_flags

            if options.build_shared_lib:
                yield self.visibility_build_flags

            if 'debug' in self.lib_flags and options.with_debug_info:
                yield process_template_string(self.lib_flags['debug'], variables, self.infofile)


        return ' '.join(list(flag_builder()))

    def gen_visibility_attribute(self, options):
        if options.build_shared_lib:
            return self.visibility_attribute
        return ''

    def ct_value_barrier_type(self, options):
        if options.ct_value_barrier_type:
            if options.ct_value_barrier_type == 'asm' and not self.supports_gcc_inline_asm:
                raise UserError('Invalid setting for --ct-value-barrier-type: the requested compiler does not support GCC inline asm')
            return options.ct_value_barrier_type

        if 'memory' in self.sanitizer_types:
            return None

        if self.ct_value_barrier:
            for pref in [options.arch, 'default']:
                if pref in self.ct_value_barrier:
                    x = self.ct_value_barrier[pref]
                    if x == 'asm' and not options.enable_asm:
                        return None
                    if x == 'none':
                        return None
                    return x

        return None

    def mach_abi_link_flags(self, options, debug_mode=None):

        """
        Return the machine specific ABI flags
        """

        if debug_mode is None:
            debug_mode = options.debug_mode

        def mach_abi_groups():

            yield 'all'

            if options.msvc_runtime is None:
                if debug_mode:
                    yield 'rt-debug'
                else:
                    yield 'rt'

            for all_except in [s for s in self.mach_abi_linking.keys() if s.startswith('all!')]:
                exceptions = all_except[4:].split(',')
                if options.os not in exceptions and options.arch not in exceptions:
                    yield all_except

            yield options.os
            yield options.arch

        abi_link = set()
        for what in mach_abi_groups():
            if what in self.mach_abi_linking:
                flag = self.mach_abi_linking.get(what)
                if flag is not None and flag != '' and flag not in abi_link:
                    abi_link.add(flag)

        if options.msvc_runtime:
            abi_link.add("/" + options.msvc_runtime)

        if options.with_stack_protector and self.stack_protector_flags != '':
            abi_link.add(self.stack_protector_flags)

        if options.with_coverage_info:
            if self.coverage_flags == '':
                raise UserError('No coverage handling for %s' % (self.basename))
            abi_link.add(self.coverage_flags)

        if options.with_sanitizers or options.enable_sanitizers != '':
            if not self.sanitizers:
                raise UserError('No sanitizer handling for %s' % (self.basename))

            default_san = self.sanitizers['default'].split(',')

            if options.enable_sanitizers:
                san = options.enable_sanitizers.split(',')
            else:
                san = default_san

            for s in san:
                if s not in self.sanitizers:
                    raise UserError('No flags defined for sanitizer %s in %s' % (s, self.basename))

                if s == 'default':
                    abi_link.update([self.sanitizers[x] for x in default_san])
                else:
                    abi_link.add(self.sanitizers[s])

            self.sanitizer_types = san

        abi_flags = ' '.join(sorted(abi_link))

        if options.cc_abi_flags != '':
            abi_flags += ' ' + options.cc_abi_flags

        return abi_flags

    def cc_warning_flags(self, options):
        def gen_flags():
            yield self.warning_flags
            if options.werror_mode or options.maintainer_mode:
                yield self.werror_flags
            if options.maintainer_mode:
                yield self.maintainer_warning_flags

        return (' '.join(gen_flags())).strip()

    def cc_lang_flags(self):
        return self.lang_flags

    def cc_lang_binary_linker_flags(self):
        return self.lang_binary_linker_flags

    def ldflags(self, options):
        if options.ldflags:
            yield options.ldflags

        if options.lto_cxxflags_to_ldflags:
            yield from self.cc_compile_flags(options)

    def cc_compile_flags(self, options):
        sanitizers_enabled = options.with_sanitizers or (len(options.enable_sanitizers) > 0)

        if options.cxxflags:
            # CXXFLAGS is assumed to be the entire set of desired compilation flags
            # if not the case the user should have used --extra-cxxflags
            yield options.cxxflags
            return

        if options.with_debug_info:
            yield self.debug_info_flags

        if not options.no_optimizations:
            if options.optimize_for_size:
                if self.size_optimization_flags != '':
                    yield self.size_optimization_flags
                else:
                    logging.warning("No size optimization flags set for current compiler")
                    yield self.optimization_flags
            elif sanitizers_enabled and self.sanitizer_optimization_flags != '':
                yield self.sanitizer_optimization_flags
            else:
                yield self.optimization_flags

        if options.arch in self.cpu_flags:
            yield self.cpu_flags[options.arch]

        if options.arch in self.cpu_flags_no_debug:
            # Only enable these if no debug/sanitizer options enabled
            if not (options.debug_mode or sanitizers_enabled):
                yield self.cpu_flags_no_debug[options.arch]

        yield from options.extra_cxxflags

        for definition in options.define_build_macro:
            yield self.add_compile_definition_option + definition

    @staticmethod
    def _so_link_search(osname, debug_info):
        so_link_typ = [osname, 'default']
        if debug_info:
            so_link_typ = [l + '-debug' for l in so_link_typ] + so_link_typ
        return so_link_typ

    def so_link_command_for(self, osname, options):
        """
        Return the command needed to link a shared object
        """

        for s in self._so_link_search(osname, options.with_debug_info):
            if s in self.so_link_commands:
                return self.so_link_commands[s]

        raise InternalError(
            "No shared library link command found for target '%s' in compiler settings '%s'" %
            (osname, self.infofile))

    def binary_link_command_for(self, osname, options):
        """
        Return the command needed to link an app/test object
        """

        for s in self._so_link_search(osname, options.with_debug_info):
            if s in self.binary_link_commands:
                return self.binary_link_commands[s]

        return '{linker}'

class OsInfo(InfoObject):
    def __init__(self, infofile):
        super().__init__(infofile)
        lex = lex_me_harder(
            infofile,
            ['aliases', 'target_features', 'feature_macros'],
            [],
            {
                'program_suffix': '',
                'obj_suffix': 'o',
                'soname_suffix': '',
                'soname_pattern_patch': '',
                'soname_pattern_abi': '',
                'soname_pattern_base': '',
                'static_suffix': 'a',
                'ar_command': 'ar',
                'ar_options': '',
                'ar_output_to': '',
                'install_root': '/usr/local',
                'header_dir': 'include',
                'bin_dir': 'bin',
                'lib_dir': 'lib',
                'doc_dir': 'share/doc',
                'man_dir': 'share/man',
                'use_stack_protector': 'true',
                'cli_exe_name': 'botan',
                'lib_prefix': 'lib',
                'library_name': 'botan-{major}{suffix}',
                'shared_lib_symlinks': 'yes',
                'default_compiler': 'gcc',
                'uses_pkg_config': 'yes',
            })

        if lex.ar_command == 'ar' and lex.ar_options == '':
            lex.ar_options = 'crs'

        if lex.soname_pattern_base:
            self.soname_pattern_base = lex.soname_pattern_base
            if lex.soname_pattern_patch == '' and lex.soname_pattern_abi == '':
                self.soname_pattern_patch = lex.soname_pattern_base
                self.soname_pattern_abi = lex.soname_pattern_base
            elif lex.soname_pattern_patch != '' and lex.soname_pattern_abi != '':
                self.soname_pattern_patch = lex.soname_pattern_patch
                self.soname_pattern_abi = lex.soname_pattern_abi
            else:
                # base set, only one of patch/abi set
                raise InternalError("Invalid soname_patterns in %s" % (self.infofile))
        else:
            if lex.soname_suffix:
                self.soname_pattern_base = "{lib_prefix}{libname}.%s" % (lex.soname_suffix)
                self.soname_pattern_abi = self.soname_pattern_base + ".{abi_rev}"
                self.soname_pattern_patch = self.soname_pattern_abi + ".{version_minor}.{version_patch}"
            else:
                # Could not calculate soname_pattern_*
                # This happens for OSs without shared library support (e.g. nacl, mingw, includeos, cygwin)
                self.soname_pattern_base = None
                self.soname_pattern_abi = None
                self.soname_pattern_patch = None

        self._aliases = lex.aliases
        self.ar_command = lex.ar_command
        self.ar_options = lex.ar_options
        self.bin_dir = lex.bin_dir
        self.cli_exe_name = lex.cli_exe_name
        self.doc_dir = lex.doc_dir
        self.header_dir = lex.header_dir
        self.install_root = lex.install_root
        self.lib_dir = lex.lib_dir
        self.lib_prefix = lex.lib_prefix
        self.library_name = lex.library_name
        self.man_dir = lex.man_dir
        self.obj_suffix = lex.obj_suffix
        self.program_suffix = lex.program_suffix
        self.static_suffix = lex.static_suffix
        self.target_features = lex.target_features
        self.use_stack_protector = lex.use_stack_protector == "true"
        self.shared_lib_uses_symlinks = lex.shared_lib_symlinks == 'yes'
        self.default_compiler = lex.default_compiler
        self.uses_pkg_config = lex.uses_pkg_config == 'yes'
        self.feature_macros = lex.feature_macros

        self._validate_os_features(self.target_features, infofile)

    @staticmethod
    def _validate_os_features(features, infofile):
        feature_re = re.compile('^[a-z][a-z0-9_]*[a-z0-9]$')
        for feature in features:
            if not feature_re.match(feature):
                logging.error("Invalid OS feature %s in %s", feature, infofile)

    def matches_name(self, nm):
        if nm in self._aliases:
            return True

        for alias in self._aliases:
            if re.match(alias, nm):
                return True
        return False

    def building_shared_supported(self):
        return self.soname_pattern_base is not None

    def enabled_features(self, options):
        feats = []
        for feat in self.target_features:
            if feat not in options.without_os_features:
                feats.append(feat)
        for feat in options.with_os_features:
            if feat not in self.target_features:
                feats.append(feat)

        return sorted(feats)

    def enabled_features_public(self, options):
        public_feat = set(['threads', 'filesystem'])
        return sorted(list(set(self.enabled_features(options)) & public_feat))

    def enabled_features_internal(self, options):
        public_feat = set(['threads', 'filesystem'])
        return sorted(list(set(self.enabled_features(options)) - public_feat))

    def macros(self, cc):
        value = [cc.add_compile_definition_option + define
                 for define in self.feature_macros]

        return ' '.join(value)

def fixup_proc_name(proc):
    proc = proc.lower().replace(' ', '')
    for junk in ['(tm)', '(r)']:
        proc = proc.replace(junk, '')
    return proc

def canon_processor(archinfo, proc):
    proc = fixup_proc_name(proc)

    # First, try to search for an exact match
    for ainfo in archinfo.values():
        if ainfo.basename == proc or proc in ainfo.aliases:
            return ainfo.basename

    return None

def system_cpu_info():

    cpu_info = []

    if platform.machine() != '':
        cpu_info.append(platform.machine())

    if platform.processor() != '':
        cpu_info.append(platform.processor())

    if 'uname' in os.__dict__:
        cpu_info.append(os.uname()[4])

    return cpu_info

def guess_processor(archinfo):
    for info_part in system_cpu_info():
        if info_part:
            match = canon_processor(archinfo, info_part)
            if match is not None:
                logging.debug("Matched '%s' to processor '%s'", info_part, match)
                return match, info_part
            else:
                logging.debug("Failed to deduce CPU from '%s'", info_part)

    raise UserError('Could not determine target CPU; set with --cpu')


def read_textfile(filepath):
    """
    Read a whole file into memory as a string
    """
    if filepath is None:
        return ''

    with open(filepath, encoding='utf8') as f:
        return ''.join(f.readlines())


def process_template_string(template_text, variables, template_source):
    """
    Perform template substitution

    The template language supports (un-nested) conditionals.
    """
    class SimpleTemplate:

        def __init__(self, vals):
            self.vals = vals
            self.value_pattern = re.compile(r'%{([a-z][a-z_0-9\|]+)(?::([^}]+))?}')
            self.cond_pattern = re.compile('%{(if|unless) ([a-z][a-z_0-9]+)}')
            self.for_pattern = re.compile('(.*)%{for ([a-z][a-z_0-9]+)}')
            self.omitlast_pattern = re.compile('(.*)%{omitlast ([^}]*)}(.*)', re.DOTALL)
            self.join_pattern = re.compile('%{join ([a-z][a-z_0-9]+)}')

        def substitute(self, template):
            def get_replacement(k):
                if k not in self.vals:
                    raise KeyError(k)
                return str(self.vals.get(k))

            def insert_value(match):
                k = match.group(1)
                if k.endswith('|upper'):
                    k = k.replace('|upper', '')
                    return get_replacement(k).upper()
                elif k.endswith('|concat'):
                    k = k.replace('|concat', '')
                    if not match.group(2):
                        raise InternalError("|concat must be of the form '%{val|concat:<some static value>}'")
                    v = get_replacement(k)
                    if v:
                        return f"{v}{match.group(2)}"
                    else:
                        return v
                elif k.endswith('|as_bool'):
                    k = k.replace('|as_bool', '')

                    if k not in self.vals:
                        raise KeyError(k)
                    v = self.vals.get(k)

                    return str(bool(v)).lower()
                else:
                    return get_replacement(k)

            def insert_join(match):
                var = match.group(1)
                if var in self.vals:
                    return ' '.join(self.vals.get(var))
                raise KeyError(var)

            lines = template.splitlines()

            output = ""
            idx = 0

            # pylint: disable=too-many-nested-blocks
            while idx < len(lines):
                cond_match = self.cond_pattern.match(lines[idx])
                for_match = self.for_pattern.match(lines[idx])

                if cond_match:
                    cond_type = cond_match.group(1)
                    cond_var = cond_match.group(2)

                    include_cond = False

                    if cond_type == 'if' and cond_var in self.vals and self.vals.get(cond_var):
                        include_cond = True
                    elif cond_type == 'unless' and (cond_var not in self.vals or (not self.vals.get(cond_var))):
                        include_cond = True

                    idx += 1
                    while idx < len(lines):
                        if lines[idx] == '%{endif}':
                            break
                        if include_cond:
                            output += lines[idx] + "\n"
                        idx += 1
                elif for_match:
                    for_prefix = for_match.group(1)
                    output += for_prefix
                    for_var = for_match.group(2)

                    if for_var not in self.vals:
                        raise InternalError("Unknown for loop iteration variable '%s'" % (for_var))

                    var = self.vals[for_var]
                    if not isinstance(var, list):
                        raise InternalError("For loop iteration variable '%s' is not a list" % (for_var))
                    idx += 1

                    for_body = ""
                    while idx < len(lines):
                        if lines[idx] == '%{endfor}':
                            break
                        for_body += lines[idx] + "\n"
                        idx += 1

                    for i, v in enumerate(var):
                        if isinstance(v, dict):
                            for_val = for_body
                            for ik, iv in v.items():
                                for_val = for_val.replace('%{' + ik + '}', iv)
                            output += for_val + "\n"
                        else:
                            output += for_body.replace('%{i}', v).replace('%{i|upper}', v.upper())

                        if output.find('%{omitlast') >= 0:
                            omitlast_match = self.omitlast_pattern.match(output)
                            if omitlast_match:
                                output = omitlast_match.group(1)
                                if i + 1 < len(var):
                                    output += omitlast_match.group(2)
                                    output += omitlast_match.group(3)

                    output += "\n"
                else:
                    output += lines[idx] + "\n"
                idx += 1

            output = self.join_pattern.sub(insert_join, self.value_pattern.sub(insert_value, output))

            # Prevent newlines being added if the template was not a multiline string
            if len(lines) == 1 and not template.endswith('\n'):
                return output.rstrip('\n')

            return output

    try:
        return SimpleTemplate(variables).substitute(template_text)
    except KeyError as ex:
        logging.error('Unbound var %s in template %s', ex, template_source)
    except Exception as ex:
        logging.error('Exception %s during template processing file %s', ex, template_source)

def process_template(template_file, variables):
    return process_template_string(read_textfile(template_file), variables, template_file)

def yield_objectfile_list(sources, obj_dir, obj_suffix, options):
    obj_suffix = '.' + obj_suffix

    for src in sources:
        (directory, filename) = os.path.split(os.path.normpath(src))
        parts_in_src = directory.split('src' + os.sep)
        parts = []

        if len(parts_in_src) > 1:
            parts = (parts_in_src[-1].split(os.sep))[1:]
        elif not options.amalgamation or filename.find(options.name_amalgamation) == -1:
            raise InternalError("Unexpected file '%s/%s'" % (directory, filename))

        if parts != []:
            # Handle src/X/X.cpp -> X.o
            if filename == parts[-1] + '.cpp':
                name = '_'.join(parts) + '.cpp'
            else:
                name = '_'.join(parts) + '_' + filename

            def fixup_obj_name(name):
                def remove_dups(parts):
                    last = None
                    for part in parts:
                        if last is None or part != last:
                            last = part
                            yield part

                return '_'.join(remove_dups(name.split('_')))

            name = fixup_obj_name(name)
        else:
            name = filename

        name = name.replace('.cpp', obj_suffix)
        yield normalize_source_path(os.path.join(obj_dir, name))

def generate_build_info(build_paths, modules, cc, arch, osinfo, options):
    # first create a map of src_file->owning module

    module_that_owns = {}

    for mod in modules:
        for src in mod.sources():
            module_that_owns[src] = mod

    def _isa_specific_flags(src):
        if os.path.basename(src) == 'test_simd.cpp':
            return cc.get_isa_specific_flags(['simd'], arch)

        if src in module_that_owns:
            module = module_that_owns[src]
            isas = module.isas_needed(arch.basename)
            if 'simd_4x32' in module.dependencies(osinfo, arch):
                isas.append('simd')

            return cc.get_isa_specific_flags(isas, arch)

        return ''

    def _build_info(sources, objects, target_type):
        output = []
        for (obj_file, src) in zip(objects, sources):
            info = {
                'src': src,
                'obj': obj_file,
                'isa_flags': _isa_specific_flags(src)
                }

            if target_type in ['fuzzer', 'examples']:
                exe_basename = os.path.basename(obj_file).replace('.' + osinfo.obj_suffix, osinfo.program_suffix)
                info['exe_basename'] = exe_basename

                if target_type == 'fuzzer':
                    info['exe'] = os.path.join(build_paths.fuzzer_output_dir, exe_basename)
                else:
                    info['exe'] = os.path.join(build_paths.example_output_dir, exe_basename)

            output.append(info)

        return output

    out = {}

    targets = ['lib', 'cli', 'test', 'fuzzer', 'examples']

    out['isa_build_info'] = []

    fuzzer_bin = []
    example_bin = []

    for t in targets:
        src_list, src_dir = build_paths.src_info(t)

        src_key = '%s_srcs' % (t)
        obj_key = '%s_objs' % (t)
        build_key = '%s_build_info' % (t)

        objects = []
        build_info = []

        if src_list is not None:
            src_list.sort()
            objects = list(yield_objectfile_list(src_list, src_dir, osinfo.obj_suffix, options))
            build_info = _build_info(src_list, objects, t)

            for b in build_info:
                if b['isa_flags'] != '':
                    out['isa_build_info'].append(b)

            if t == 'fuzzer':
                fuzzer_bin = [b['exe'] for b in build_info]
            elif t == 'examples':
                example_bin = [b['exe'] for b in build_info]

        out[src_key] = src_list if src_list else []
        out[obj_key] = objects
        out[build_key] = build_info

    out['fuzzer_bin'] = ' '.join(fuzzer_bin)
    out['example_bin'] = ' '.join(example_bin)
    out['cli_headers'] = build_paths.cli_headers

    return out

def create_template_vars(source_paths, build_paths, options, modules, disabled_modules, cc, arch, osinfo):
    """
    Create the template variables needed to process the makefile, build.h, etc
    """

    def external_link_cmd():
        return ' '.join([cc.add_lib_dir_option + libdir for libdir in options.with_external_libdir])

    def link_to(module_member_name):
        """
        Figure out what external libraries/frameworks are needed based on selected modules
        """
        if module_member_name not in ['libs', 'frameworks']:
            raise InternalError("Invalid argument")

        libs = set()
        for module in modules:
            for (osname, module_link_to) in getattr(module, module_member_name).items():
                if osname in ['all', osinfo.basename]:
                    libs |= set(module_link_to)
                else:
                    match = re.match('^all!(.*)', osname)
                    if match is not None:
                        exceptions = match.group(1).split(',')
                        if osinfo.basename not in exceptions:
                            libs |= set(module_link_to)

        return sorted(libs)

    def configure_command_line():
        # Cut absolute path from main executable (e.g. configure.py or python interpreter)
        # to get the same result when configuring the same thing on different machines
        main_executable = os.path.basename(sys.argv[0])
        quoted_args = [arg if ' ' not in arg else '\'' + arg + '\'' for arg in sys.argv[1:]]
        return ' '.join([main_executable] + quoted_args)

    def sysroot_option():
        if options.with_sysroot_dir == '':
            return ''
        if cc.add_sysroot_option == '':
            logging.error("This compiler doesn't support --sysroot option")
        return cc.add_sysroot_option + options.with_sysroot_dir

    def ar_command():
        if options.ar_command:
            return options.ar_command

        if cc.ar_command:
            if cc.ar_command == cc.binary_name:
                return options.compiler_binary or cc.binary_name
            else:
                return cc.ar_command

        return osinfo.ar_command

    build_dir = options.with_build_dir or os.path.curdir
    program_suffix = options.program_suffix or osinfo.program_suffix

    def join_with_build_dir(path):
        # jom (and mingw32-make) seem to string-compare Makefile targets and
        # requirements. For them, `./botan.lib` is NOT equal to `botan.lib` or
        # `C:\botan\botan-test.exe` is NOT equal to `C:\botan/botan-test.exe`
        #
        # `normalize_source_path` will "fix" the path slashes but remove
        # a redundant `./` for the "trivial" relative path.
        normalized = normalize_source_path(os.path.join(build_dir, path))
        if build_dir == '.':
            normalized = './%s' % normalized
        return normalized

    def all_targets(options):
        yield 'libs'
        if options.with_documentation:
            yield 'docs'
        if 'cli' in options.build_targets:
            yield 'cli'
        if 'tests' in options.build_targets:
            yield 'tests'
        if options.build_fuzzers:
            yield 'fuzzers'
        if 'bogo_shim' in options.build_targets:
            yield 'bogo_shim'
        if 'examples' in options.build_targets:
            yield 'examples'
        if 'ct_selftest' in options.build_targets:
            yield 'ct_selftest'

    def install_targets(options):
        yield 'libs'
        if 'cli' in options.build_targets:
            yield 'cli'
        if options.with_documentation:
            yield 'docs'

    def absolute_install_dir(p):
        if os.path.isabs(p):
            return p
        return os.path.join(options.prefix or osinfo.install_root, p)

    def choose_python_exe():
        return normalize_source_path(sys.executable)

    def choose_cxx_exe(with_compiler_cache = False):
        cxx = options.compiler_binary or cc.binary_name

        if options.compiler_cache is None or with_compiler_cache is False:
            return cxx
        else:
            return '%s %s' % (normalize_source_path(options.compiler_cache), cxx)

    def extra_libs(libs, cc):
        if libs is None:
            return ''

        return ' '.join([(cc.add_lib_option % lib) for lib in libs.split(',') if lib != ''])

    def test_exe_extra_ldflags():
        if osinfo.matches_name("emscripten"):
            return '--preload-file=%s@src/tests/data' % source_paths.test_data_dir

        return ''

    variables = {
        'version_major':  Version.major(),
        'version_minor':  Version.minor(),
        'version_patch':  Version.patch(),
        'version_vc_rev': None if Version.vc_rev() == 'unknown' else Version.vc_rev(),

        'version_vc_rev_or_unknown': 'unknown' if Version.datestamp() == 0 else Version.vc_rev(),

        'abi_rev':        Version.so_rev(),

        'version':        Version.as_string(),
        'release_type':   Version.release_type(),
        'version_datestamp': Version.datestamp(),

        'distribution_info': options.distribution_info,
        'distribution_info_or_unspecified': options.distribution_info or 'unspecified',

        'full_version_string': Version.full_version_string(options),
        'short_version_string': Version.short_version_string(),

        'macos_so_compat_ver': '%s.%s.0' % (Version.packed(), Version.so_rev()),
        'macos_so_current_ver': '%s.%s.%s' % (Version.packed(), Version.so_rev(), Version.patch()),

        'all_targets': ' '.join(all_targets(options)),
        'install_targets': ' '.join(install_targets(options)),

        'public_headers': sorted([os.path.basename(h) for h in build_paths.public_headers]),
        'internal_headers': sorted([os.path.basename(h) for h in build_paths.internal_headers]),
        'external_headers':  sorted([os.path.basename(h) for h in build_paths.external_headers]),

        'abs_root_dir': normalize_source_path(os.path.dirname(os.path.realpath(__file__))),

        'base_dir': source_paths.base_dir,
        'src_dir': source_paths.src_dir,
        'test_data_dir': source_paths.test_data_dir,
        'doc_dir': source_paths.doc_dir,
        'scripts_dir': normalize_source_path(source_paths.scripts_dir),
        'python_dir': source_paths.python_dir,

        'cli_exe_name': osinfo.cli_exe_name + program_suffix,
        'cli_exe': join_with_build_dir(osinfo.cli_exe_name + program_suffix),
        'build_cli_exe': bool('cli' in options.build_targets),
        'test_exe': join_with_build_dir('botan-test' + program_suffix),

        'lib_prefix': osinfo.lib_prefix,
        'static_suffix': osinfo.static_suffix,
        'lib_suffix': options.library_suffix,
        'libname': osinfo.library_name.format(major=Version.major(),
                                              minor=Version.minor(),
                                              suffix=options.library_suffix),

        'command_line': configure_command_line(),

        'program_suffix': program_suffix,

        'prefix': options.prefix or osinfo.install_root,
        'bindir': absolute_install_dir(options.bindir or osinfo.bin_dir),
        'libdir': absolute_install_dir(options.libdir or osinfo.lib_dir),
        'mandir': options.mandir or osinfo.man_dir,
        'includedir': options.includedir or osinfo.header_dir,
        'docdir': options.docdir or osinfo.doc_dir,

        'with_documentation': options.with_documentation,
        'with_sphinx': options.with_sphinx,
        'with_pdf': options.with_pdf,
        'with_rst2man': options.with_rst2man,
        'sphinx_config_dir': source_paths.sphinx_config_dir,
        'with_doxygen': options.with_doxygen,
        'maintainer_mode': options.maintainer_mode,

        'out_dir': normalize_source_path(build_dir),
        'build_dir': normalize_source_path(build_paths.build_dir),
        'module_info_dir': build_paths.doc_module_info,

        'doc_stamp_file': normalize_source_path(os.path.join(build_paths.build_dir, 'doc.stamp')),
        'makefile_path': os.path.join(build_paths.build_dir, '..', 'Makefile'),
        'ninja_build_path': os.path.join(build_paths.build_dir, '..', 'build.ninja'),

        # Use response files for the archive command on windows
        # Note: macOS (and perhaps other OSes) do not support this
        'build_static_lib_using_cmdline_args': options.build_static_lib and osinfo.basename != 'windows',
        'build_static_lib_using_response_file': options.build_static_lib and osinfo.basename == 'windows',
        'build_static_lib': options.build_static_lib,
        'build_shared_lib': options.build_shared_lib,

        'build_fuzzers': options.build_fuzzers,
        'build_examples': 'examples' in options.build_targets,

        'build_coverage' : options.with_coverage_info,

        'symlink_shared_lib': options.build_shared_lib and osinfo.shared_lib_uses_symlinks,

        'libobj_dir': build_paths.libobj_dir,
        'cliobj_dir': build_paths.cliobj_dir,
        'testobj_dir': build_paths.testobj_dir,
        'fuzzobj_dir': build_paths.fuzzobj_dir,

        'fuzzer_output_dir': build_paths.fuzzer_output_dir if build_paths.fuzzer_output_dir else '',
        'doc_output_dir': build_paths.doc_output_dir,
        'handbook_output_dir': build_paths.handbook_output_dir,
        'doc_output_dir_doxygen': build_paths.doc_output_dir_doxygen,
        'response_file_dir': build_paths.response_file_dir,

        'os': options.os,
        'arch': options.arch,
        'compiler': options.compiler,
        'cpu_family': arch.family,

        'python_exe': choose_python_exe(),
        'python_version': options.python_version,
        'install_python_module': not options.no_install_python_module,

        'cxx': choose_cxx_exe(True),
        'cxx_abi_flags': cc.mach_abi_link_flags(options),
        'linker': cc.linker_name or choose_cxx_exe(),
        'make_supports_phony': osinfo.basename != 'windows',

        'cxx_supports_gcc_inline_asm': cc.supports_gcc_inline_asm and options.enable_asm,
        'compiler_assisted_stack_scrubbing': options.enable_stack_scrubbing,

        'cxx_ct_value_barrier_type': cc.ct_value_barrier_type(options),

        'sanitizer_types' : sorted(cc.sanitizer_types),

        'dash_o': cc.output_to_object,
        'dash_c': cc.compile_flags,

        'cc_lang_flags': cc.cc_lang_flags(),
        'cc_lang_binary_linker_flags': cc.cc_lang_binary_linker_flags(),
        'os_feature_macros': osinfo.macros(cc),
        'cc_sysroot': sysroot_option(),
        'cc_compile_flags': ' '.join(cc.cc_compile_flags(options)).strip(),
        'ldflags': ' '.join(cc.ldflags(options)).strip(),
        'test_exe_extra_ldflags': test_exe_extra_ldflags(),
        'extra_libs': extra_libs(options.extra_libs, cc),
        'cc_warning_flags': cc.cc_warning_flags(options),
        'output_to_exe': cc.output_to_exe,
        'cc_macro': cc.macro_name,
        'ninja_header_deps_style': cc.ninja_header_deps_style,
        'header_deps_flag': cc.header_deps_flag,
        'header_deps_out': cc.header_deps_out,

        'visibility_attribute': cc.gen_visibility_attribute(options),

        'lib_link_cmd': cc.so_link_command_for(osinfo.basename, options),
        'exe_link_cmd': cc.binary_link_command_for(osinfo.basename, options),
        'external_link_cmd': external_link_cmd(),

        'ar_command': ar_command(),
        'ar_options': options.ar_options or cc.ar_options or osinfo.ar_options,
        'ar_output_to': cc.ar_output_to,

        'link_to': ' '.join(
            [(cc.add_lib_option % lib) for lib in link_to('libs')] +
            [cc.add_framework_option + fw for fw in link_to('frameworks')]
        ),

        'fuzzer_lib': (cc.add_lib_option % options.fuzzer_lib) if options.fuzzer_lib else '',
        'libs_used': [lib.replace('.lib', '') for lib in link_to('libs')],

        'public_include_path': build_paths.public_include_dir,
        'internal_include_path': build_paths.internal_include_dir,
        'external_include_path': build_paths.external_include_dir,

        'public_include_flags': build_paths.format_public_include_flags(cc),
        'internal_include_flags': build_paths.format_internal_include_flags(cc),
        'external_include_flags': build_paths.format_external_include_flags(cc, options.with_external_includedir),
        'module_defines': sorted(flatten([m.defines() for m in modules])),
        'module_internal_defines': sorted(flatten([m.internal_defines() for m in modules])),

        'build_bogo_shim': bool('bogo_shim' in options.build_targets),
        'bogo_shim_src': os.path.join(source_paths.src_dir, 'bogo_shim', 'bogo_shim.cpp'),

        'build_ct_selftest': bool('ct_selftest' in options.build_targets),
        'ct_selftest_src': os.path.join(source_paths.src_dir, 'ct_selftest', 'ct_selftest.cpp'),

        'os_features': osinfo.enabled_features_internal(options),
        'os_features_public': osinfo.enabled_features_public(options),
        'os_name': osinfo.basename,
        'cpu_features': arch.supported_isa_extensions(cc),
        'system_cert_bundle': options.system_cert_bundle,

        'enable_experimental_features': options.enable_experimental_features,
        'disable_deprecated_features': not options.enable_deprecated_features,

        'fuzzer_mode': options.unsafe_fuzzer_mode,
        'building_fuzzers': options.build_fuzzers,
        'fuzzer_type': options.build_fuzzers.upper() if options.build_fuzzers else '',

        'with_valgrind': options.with_valgrind,
        'with_debug_asserts': options.with_debug_asserts,
        'terminate_on_asserts': options.unsafe_terminate_on_asserts,
        'optimize_for_size': options.optimize_for_size,

        'mod_list': sorted([m.basename for m in modules]),
        'disabled_mod_list': sorted([m.basename for m in disabled_modules]),
    }

    variables['installed_include_dir'] = os.path.join(
        variables['prefix'],
        variables['includedir'],
        'botan-%d' % (Version.major()), 'botan')

    if cc.basename == 'msvc' and variables['cxx_abi_flags'] != '':
        # MSVC linker doesn't support/need the ABI options,
        # just transfer them over to just the compiler invocations
        variables['cc_compile_flags'] = '%s %s' % (variables['cxx_abi_flags'], variables['cc_compile_flags'])
        variables['cxx_abi_flags'] = ''

    variables['lib_flags'] = cc.gen_lib_flags(options, variables)

    if options.with_pkg_config:
        variables['botan_pkgconfig'] = os.path.join(build_paths.build_dir, 'botan-%d.pc' % (Version.major()))
    if options.with_cmake_config:
        variables['botan_cmake_config'] = os.path.join(build_paths.build_dir, 'cmake', 'botan-config.cmake')
        variables['botan_cmake_version_config'] = os.path.join(build_paths.build_dir, 'cmake', 'botan-config-version.cmake')

    # The name is always set because Windows build needs it
    variables['static_lib_name'] = '%s%s.%s' % (variables['lib_prefix'], variables['libname'],
                                                variables['static_suffix'])

    if options.build_shared_lib:
        if osinfo.soname_pattern_base is not None:
            variables['soname_base'] = osinfo.soname_pattern_base.format(**variables)
            variables['shared_lib_name'] = variables['soname_base']

        if osinfo.soname_pattern_abi is not None:
            variables['soname_abi'] = osinfo.soname_pattern_abi.format(**variables)
            variables['shared_lib_name'] = variables['soname_abi']

        if osinfo.soname_pattern_patch is not None:
            variables['soname_patch'] = osinfo.soname_pattern_patch.format(**variables)

        if options.os == 'windows':
            variables['implib_name'] = variables['static_lib_name']

        variables['lib_link_cmd'] = variables['lib_link_cmd'].format(**variables)

    for var in ['exe_link_cmd']:
        variables[var] = variables[var].format(**variables)

    lib_targets = []
    if options.build_static_lib:
        lib_targets.append('static_lib_name')
    if options.build_shared_lib:
        lib_targets.append('shared_lib_name')

    variables['library_targets'] = ' '.join([join_with_build_dir(variables[t]) for t in lib_targets])

    if options.os == 'llvm' or options.compiler == 'msvc':
        # llvm-link and msvc require just naming the file directly
        variables['build_dir_link_path'] = ''
        variables['link_to_botan'] = normalize_source_path(os.path.join(build_dir, variables['static_lib_name']))
    else:
        variables['build_dir_link_path'] = '%s%s' % (cc.add_lib_dir_option, build_dir)
        variables['link_to_botan'] = cc.add_lib_option % variables['libname']

    return variables

class ModulesChooser:
    """
    Determine which modules to load based on options, target, etc
    """

    def __init__(self, modules, module_policy, archinfo, osinfo, ccinfo, cc_min_version, options):
        self._modules = modules
        self._module_policy = module_policy
        self._archinfo = archinfo
        self._osinfo = osinfo
        self._ccinfo = ccinfo
        self._cc_min_version = cc_min_version
        self._options = options

        self._maybe_dep = set()
        self._to_load = set()
        # string to set mapping with reasons as key and modules as value
        self._not_using_because = collections.defaultdict(set)

        ModulesChooser._validate_dependencies_exist(self._modules)
        self._options.enabled_modules = ModulesChooser._expand_wildcards_in_user_selection(self._modules, self._options.enabled_modules)
        self._options.disabled_modules = ModulesChooser._expand_wildcards_in_user_selection(self._modules, self._options.disabled_modules)
        ModulesChooser._validate_user_selection(
            self._modules, self._options.enabled_modules, self._options.disabled_modules)

    def _check_usable(self, module, modname):
        if not module.compatible_cpu(self._archinfo, self._options):
            self._not_using_because['incompatible CPU'].add(modname)
            return False
        elif not module.compatible_os(self._osinfo, self._options):
            self._not_using_because['incompatible OS'].add(modname)
            return False
        elif not module.compatible_compiler(self._ccinfo, self._cc_min_version, self._archinfo.basename):
            self._not_using_because['incompatible compiler'].add(modname)
            return False
        elif module.is_deprecated() and not self._options.enable_deprecated_features and modname not in self._options.enabled_modules:
            self._not_using_because['deprecated'].add(modname)
            return False
        elif module.is_experimental() and modname not in self._options.enabled_modules and not self._options.enable_experimental_features:
            self._not_using_because['experimental'].add(modname)
            return False
        return True

    @staticmethod
    def _remove_virtual_modules(all_modules, modnames):
        return [mod for mod in modnames if not all_modules[mod].is_virtual()]

    @classmethod
    def _display_module_information_unused(cls, all_modules, skipped_modules):
        for reason in sorted(skipped_modules.keys()):
            disabled_mods = cls._remove_virtual_modules(all_modules, sorted(skipped_modules[reason]))
            if disabled_mods:
                logging.info('Skipping (%s): %s', reason, ' '.join(disabled_mods))

    @classmethod
    def _display_module_information_to_load(cls, all_modules, modules_to_load):
        sorted_modules_to_load = cls._remove_virtual_modules(all_modules, sorted(modules_to_load))

        deprecated = []
        experimental = []
        for modname in sorted_modules_to_load:
            if all_modules[modname].comment:
                logging.info('%s: %s', modname, all_modules[modname].comment)
            if all_modules[modname].warning:
                logging.warning('%s: %s', modname, all_modules[modname].warning)
            if all_modules[modname].load_on == 'vendor':
                logging.info('Enabling use of external dependency %s', modname)
            if all_modules[modname].is_deprecated():
                deprecated.append(modname)
            if all_modules[modname].is_experimental():
                experimental.append(modname)

        if deprecated:
            logging.warning('These modules are deprecated and will be removed in a future release (consider disabling with --disable-deprecated-features): %s',
                            ' '.join(deprecated))

        if experimental:
            logging.warning('These modules are experimental and may change or be removed in a future release: %s',
                            ' '.join(experimental))

        if sorted_modules_to_load:
            logging.info('Loading modules: %s', ' '.join(sorted_modules_to_load))
        else:
            logging.error('This configuration disables every submodule and is invalid')

    @staticmethod
    def _validate_state(used_modules, unused_modules):
        for reason, unused_for_reason in unused_modules.items():
            intersection = unused_for_reason & used_modules
            if intersection:
                raise InternalError(
                    "Disabled modules (%s) and modules to load have common elements: %s"
                    % (reason, intersection))

    @staticmethod
    def _validate_dependencies_exist(modules):
        for module in modules.values():
            module.dependencies_exist(modules)

    @staticmethod
    def _expand_wildcards_in_user_selection(modules, user_selected_modules):
        valid_module_name_with_wildcard = re.compile(r'^[a-z0-9_*]+$')
        public_modules = [modname for modname, modinfo in modules.items() if modinfo.is_public()]
        def expand(user_selected_module):
            if not valid_module_name_with_wildcard.match(user_selected_module):
                logging.error("Invalid module name with wildcard: %s", user_selected_module)
                return []
            regex_from_wildcards = re.compile("^%s$" % user_selected_module.replace('*', '[a-z0-9_]+'))
            matching_modules = [mod for mod in public_modules if regex_from_wildcards.match(mod)]
            if not matching_modules:
                logging.warning("Wildcard '%s' did not match any modules", user_selected_module)
            return matching_modules
        return flatten([expand(mod) if '*' in mod else [mod] for mod in user_selected_modules])

    @staticmethod
    def _validate_user_selection(modules, enabled_modules, disabled_modules):
        for modname in enabled_modules:
            if modname not in modules:
                logging.error("Module not found: %s", modname)
            if not modules[modname].is_public():
                logging.error("Module '%s' is meant for internal use only", modname)

        for modname in disabled_modules:
            if modname not in modules:
                logging.warning("Disabled module not found: %s", modname)

    def _handle_by_module_policy(self, modname, module, usable):
        if self._module_policy is not None:
            if modname in self._module_policy.required:
                if not usable:
                    logging.error('Module policy requires module %s not usable on this platform', modname)
                elif modname in self._options.disabled_modules:
                    logging.error('Module %s was disabled but is required by policy', modname)
                elif module.is_virtual():
                    logging.error("Module %s is meant for internal use only", modname)
                self._to_load.add(modname)
                return True
            elif modname in self._module_policy.if_available:
                if modname in self._options.disabled_modules:
                    self._not_using_because['disabled by user'].add(modname)
                elif module.is_virtual():
                    logging.error("Module %s is meant for internal use only", modname)
                elif usable:
                    logging.debug('Enabling optional module %s', modname)
                    self._to_load.add(modname)
                return True
            elif modname in self._module_policy.prohibited:
                if modname in self._options.enabled_modules:
                    logging.error('Module %s was requested but is prohibited by policy', modname)
                self._not_using_because['prohibited by module policy'].add(modname)
                return True

        return False

    @staticmethod
    def resolve_dependencies(available_modules, dependency_table, module, loaded_modules=None):
        """
        Parameters
        - available_modules: modules to choose from. Constant.
        - dependency_table: module to dependencies map. Constant.
        - module: name of the module to resolve dependencies. Constant.
        - loaded_modules: modules already loaded. Defensive copy in order to not change value for caller.
        """
        if loaded_modules is None:
            loaded_modules = set([])
        else:
            loaded_modules = copy.copy(loaded_modules)

        if module not in available_modules:
            return False, None

        loaded_modules.add(module)
        for dependency in dependency_table[module]:
            dependency_choices = set(dependency.split('|'))

            dependency_met = False

            if not set(dependency_choices).isdisjoint(loaded_modules):
                dependency_met = True
            else:
                possible_mods = dependency_choices.intersection(available_modules)

                for mod in possible_mods:
                    ok, dependency_modules = ModulesChooser.resolve_dependencies(
                        available_modules, dependency_table, mod, loaded_modules)
                    if ok:
                        dependency_met = True
                        loaded_modules.add(mod)
                        loaded_modules.update(dependency_modules)
                        break

            if not dependency_met:
                return False, None

        return True, loaded_modules

    def _modules_dependency_table(self):
        out = {}
        for modname in self._modules:
            out[modname] = self._modules[modname].dependencies(self._osinfo, self._archinfo)
        return out

    def _resolve_dependencies_for_all_modules(self):
        available_modules = set(self._to_load) | set(self._maybe_dep)
        dependency_table = self._modules_dependency_table()

        successfully_loaded = set()

        for modname in self._to_load:
            # This will try to recursively load all dependencies of modname
            ok, modules = self.resolve_dependencies(available_modules, dependency_table, modname)
            if ok:
                successfully_loaded.add(modname)
                successfully_loaded.update(modules)
            else:
                # Skip this module
                pass

        self._not_using_because['dependency failure'].update(self._to_load - successfully_loaded)
        self._to_load = successfully_loaded
        self._maybe_dep -= successfully_loaded

    def _handle_by_load_on(self, module):
        modname = module.basename
        if module.load_on == 'never':
            self._not_using_because['disabled as buggy'].add(modname)
        elif module.load_on == 'request':
            self._not_using_because['by request only'].add(modname)
        elif module.load_on == 'vendor':
            self._not_using_because['requires external dependency'].add(modname)
        elif module.load_on == 'dep':
            self._maybe_dep.add(modname)

        elif module.load_on == 'always':
            self._to_load.add(modname)

        elif module.load_on == 'auto':
            if self._options.no_autoload or self._module_policy is not None:
                self._maybe_dep.add(modname)
            else:
                self._to_load.add(modname)
        else:
            logging.error('Unknown load_on %s in %s',
                          module.load_on, modname)

    def choose(self):
        for (modname, module) in self._modules.items():
            usable = self._check_usable(module, modname)

            module_handled = self._handle_by_module_policy(modname, module, usable)
            if module_handled:
                continue

            if modname in self._options.disabled_modules:
                self._not_using_because['disabled by user'].add(modname)
            elif usable:
                if modname in self._options.enabled_modules:
                    self._to_load.add(modname) # trust the user
                else:
                    self._handle_by_load_on(module)

        if 'compression' in self._to_load:
            # Confirm that we have at least one compression library enabled
            # Otherwise we leave a lot of useless support code compiled in, plus a
            # make_compressor call that always fails
            if 'zlib' not in self._to_load and 'bzip2' not in self._to_load and 'lzma' not in self._to_load:
                self._to_load.remove('compression')
                self._not_using_because['no enabled compression schemes'].add('compression')

        # The AVX2 implementation of Argon2 fails when compiled by GCC in
        # amalgamation mode.
        if 'argon2_avx2' in self._to_load and self._options.amalgamation and self._options.compiler == 'gcc':
            self._to_load.remove('argon2_avx2')
            self._not_using_because['disabled due to compiler bug'].add('argon2_avx2')

        self._resolve_dependencies_for_all_modules()

        for not_a_dep in self._maybe_dep:
            self._not_using_because['not requested'].add(not_a_dep)

        ModulesChooser._validate_state(self._to_load, self._not_using_because)
        ModulesChooser._display_module_information_unused(self._modules, self._not_using_because)
        ModulesChooser._display_module_information_to_load(self._modules, self._to_load)

        return self._to_load

def choose_link_method(options):
    """
    Choose the link method based on system availability and user request
    """

    req = options.link_method

    def useable_methods():

        # Symbolic link support on Windows was introduced in Windows 6.0 (Vista)
        # and Python 3.2. Furthermore, the SeCreateSymbolicLinkPrivilege is
        # required in order to successfully create symlinks. So only try to use
        # symlinks on Windows if explicitly requested.

        # Hardlinks only work if the source and build dirs are on the same filesystem,
        # so there we only use it if requested.

        # MinGW declares itself as 'Windows'
        host_is_windows = python_platform_identifier() in ['windows', 'cygwin']

        if 'symlink' in os.__dict__:
            if host_is_windows:
                if req == 'symlink':
                    yield 'symlink'
            else:
                yield 'symlink'

        if 'link' in os.__dict__ and req == 'hardlink':
            yield 'hardlink'

        yield 'copy'

    for method in useable_methods():
        if req is None or req == method:
            logging.info('Using %s to link files into build dir ' \
                         '(use --link-method to change)', method)
            return method

    logging.warning('Could not use link method "%s", will copy instead', req)
    return 'copy'

def portable_symlink(file_path, target_dir, method):
    """
    Copy or link the file, depending on what the platform offers
    """

    if not os.access(file_path, os.R_OK):
        logging.warning('Missing file %s', file_path)
        return

    if method == 'symlink':
        rel_file_path = os.path.relpath(file_path, start=target_dir)
        os.symlink(rel_file_path, os.path.join(target_dir, os.path.basename(file_path)))
    elif method == 'hardlink':
        os.link(file_path, os.path.join(target_dir, os.path.basename(file_path)))
    elif method == 'copy':
        shutil.copy(file_path, target_dir)
    else:
        raise UserError('Unknown link method %s' % (method))


class AmalgamationHelper:
    # All include types may have trailing comment like e.g. '#include <vector> // IWYU pragma: export'
    _any_include = re.compile(r'#include <(.*)>')
    _botan_include = re.compile(r'#include <botan/(.*)>')

    # Only matches at the beginning of the line. By convention, this means that the include
    # is not wrapped by condition macros
    _unconditional_any_include = re.compile(r'^#include <(.*)>')
    # stddef.h is included in ffi.h
    _unconditional_std_include = re.compile(r'^#include <([^/\.]+|stddef.h)>')

    @staticmethod
    def is_any_include(cpp_source_line):
        match = AmalgamationHelper._any_include.search(cpp_source_line)
        if match:
            return match.group(1)
        else:
            return None

    @staticmethod
    def is_botan_include(cpp_source_line):
        match = AmalgamationHelper._botan_include.search(cpp_source_line)
        return match.group(1) if match else None

    @staticmethod
    def is_unconditional_any_include(cpp_source_line):
        match = AmalgamationHelper._unconditional_any_include.search(cpp_source_line)
        return match.group(1) if match else None

    @staticmethod
    def is_unconditional_std_include(cpp_source_line):
        match = AmalgamationHelper._unconditional_std_include.search(cpp_source_line)
        return match.group(1) if match else None

    @staticmethod
    def write_banner(fd):
        fd.write("""/*
* Botan %s Amalgamation
* (C) 1999-2023 The Botan Authors
*
* Botan is released under the Simplified BSD License (see license.txt)
*/
""" % (Version.as_string()))


class AmalgamationHeader:
    def __init__(self, input_filepaths):

        self.included_already = set()
        self.all_std_includes = set()

        self.file_contents = {}
        for filepath in sorted(input_filepaths):
            try:
                contents = AmalgamationGenerator.read_header(filepath)
                self.file_contents[os.path.basename(filepath)] = contents
            except IOError as ex:
                logging.error('Error processing file %s for amalgamation: %s', filepath, ex)

        self.contents = ''
        for name in sorted(self.file_contents):
            self.contents += ''.join(list(self.header_contents(name)))

        self.header_includes = ''
        for std_header in sorted(self.all_std_includes):
            self.header_includes += '#include <%s>\n' % (std_header)
        self.header_includes += '\n'

    def header_contents(self, name):
        name = name.replace('internal/', '')

        if name in self.included_already:
            return

        self.included_already.add(name)

        if name not in self.file_contents:
            return

        depr_marker = 'BOTAN_DEPRECATED_HEADER(%s)\n' % (name)
        if depr_marker in self.file_contents[name]:
            logging.debug("Ignoring deprecated header %s", name)
            return

        for line in self.file_contents[name]:
            header = AmalgamationHelper.is_botan_include(line)
            if header:
                yield from self.header_contents(header)
            else:
                std_header = AmalgamationHelper.is_unconditional_std_include(line)

                if std_header:
                    self.all_std_includes.add(std_header)
                else:
                    yield line

    def write_to_file(self, filepath, include_guard):
        with open(filepath, 'w', encoding='utf8') as f:
            AmalgamationHelper.write_banner(f)
            f.write("\n#ifndef %s\n#define %s\n\n" % (include_guard, include_guard))
            f.write(self.header_includes)
            f.write(self.contents)
            f.write("\n#endif // %s\n" % (include_guard))


class AmalgamationGenerator:
    _header_guard_pattern = re.compile(r'^#define BOTAN_.*_H_\s*$')
    _header_endif_pattern = re.compile(r'^#endif.*$')

    @staticmethod
    def read_header(filepath):
        with open(filepath, encoding='utf8') as f:
            raw_content = f.readlines()
            return AmalgamationGenerator.strip_header_goop(filepath, raw_content)

    @staticmethod
    def strip_header_goop(header_name, header_lines):
        lines = copy.copy(header_lines) # defensive copy

        start_header_guard_index = None
        for index, line in enumerate(lines):
            if AmalgamationGenerator._header_guard_pattern.match(line):
                start_header_guard_index = index
                break
        if start_header_guard_index is None:
            raise InternalError("No header guard start found in " + header_name)

        end_header_guard_index = None
        for index, line in enumerate(lines):
            if AmalgamationGenerator._header_endif_pattern.match(line):
                end_header_guard_index = index # override with last found
        if end_header_guard_index is None:
            raise InternalError("No header guard end found in " + header_name)

        lines = lines[start_header_guard_index+1 : end_header_guard_index]

        # Strip leading and trailing empty lines
        while lines[0].strip() == "":
            lines = lines[1:]
        while lines[-1].strip() == "":
            lines = lines[0:-1]

        return lines

    def __init__(self, prefix, build_paths, modules, options):
        self._filename_prefix = prefix
        self._build_paths = build_paths
        self._modules = modules
        self._options = options

    def generate(self):
        pub_header_amalag = AmalgamationHeader(self._build_paths.public_headers)
        amalgamation_header_fsname = '%s.h' % (self._filename_prefix)
        logging.info('Writing amalgamation header to %s', amalgamation_header_fsname)
        pub_header_amalag.write_to_file(amalgamation_header_fsname, "BOTAN_AMALGAMATION_H_")

        internal_headers_list = []

        for hdr in self._build_paths.internal_headers:
            internal_headers_list.append(hdr)

        # file descriptors for all `amalgamation_sources`
        amalgamation_fsname = '%s.cpp' % (self._filename_prefix)
        logging.info('Writing amalgamation source to %s', amalgamation_fsname)

        amalgamation_file = open(amalgamation_fsname, 'w', encoding='utf8')

        AmalgamationHelper.write_banner(amalgamation_file)
        amalgamation_file.write('\n#include "%s"\n\n' % (amalgamation_header_fsname))

        internal_headers = AmalgamationHeader(internal_headers_list)
        amalgamation_file.write(internal_headers.header_includes)
        amalgamation_file.write(internal_headers.contents)

        unconditional_headers = set([])

        for mod in sorted(self._modules, key=lambda module: module.basename):
            for src in sorted(mod.source):
                with open(src, 'r', encoding='utf8') as f:
                    for line in f:
                        if AmalgamationHelper.is_botan_include(line):
                            # Botan headers are inlined in amalgamation headers
                            continue

                        if AmalgamationHelper.is_any_include(line) in unconditional_headers:
                            # This include (conditional or unconditional) was unconditionally added before
                            continue

                        amalgamation_file.write(line)
                        unconditional_header = AmalgamationHelper.is_unconditional_any_include(line)
                        if unconditional_header:
                            unconditional_headers.add(unconditional_header)

        amalgamation_file.close()

        return ([amalgamation_fsname], [amalgamation_header_fsname])


def have_program(program):
    """
    Test for the existence of a program
    """

    def exe_test(path, program):
        exe_file = os.path.join(path, program)

        if os.path.exists(exe_file) and os.access(exe_file, os.X_OK):
            logging.debug('Found program %s in %s', program, path)
            return True
        else:
            return False

    exe_suffixes = ['', '.exe']

    for path in os.environ['PATH'].split(os.pathsep):
        for suffix in exe_suffixes:
            if exe_test(path, program + suffix):
                return True

    logging.debug('Program %s not found', program)
    return False


class BotanConfigureLogHandler(logging.StreamHandler):
    def emit(self, record):
        # Do the default stuff first
        super().emit(record)
        # Exit script if and ERROR or worse occurred
        if record.levelno >= logging.ERROR:
            sys.exit(1)


def setup_logging(options):
    if options.verbose:
        log_level = logging.DEBUG
    elif options.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    lh = BotanConfigureLogHandler(sys.stdout)
    lh.setFormatter(logging.Formatter('%(levelname) 7s: %(message)s'))
    logging.getLogger().addHandler(lh)
    logging.getLogger().setLevel(log_level)


def load_info_files(search_dir, descr, filename_matcher, class_t):
    info = {}

    def filename_matches(filename):
        if isinstance(filename_matcher, str):
            return filename == filename_matcher
        else:
            return filename_matcher.match(filename) is not None

    for (dirpath, _, filenames) in os.walk(search_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if filename_matches(filename):
                info_obj = class_t(filepath)
                info[info_obj.basename] = info_obj

    if info:
        infotxt_basenames = ' '.join(sorted(info.keys()))
        logging.debug('Loaded %d %s files: %s', len(info), descr, infotxt_basenames)
    else:
        logging.warning('Failed to load any %s files', descr)

    return info


def load_build_data_info_files(source_paths, descr, subdir, class_t):
    matcher = re.compile(r'[_a-z0-9]+\.txt$')
    return load_info_files(os.path.join(source_paths.build_data_dir, subdir), descr, matcher, class_t)


# Workaround for Windows systems where antivirus is enabled GH #353
def robust_rmtree(path, max_retries=5):
    for _ in range(max_retries):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            time.sleep(0.1)

    # Final attempt, pass any exceptions up to caller.
    shutil.rmtree(path)


# Workaround for Windows systems where antivirus is enabled GH #353
def robust_makedirs(directory, max_retries=5):
    for _ in range(max_retries):
        try:
            os.makedirs(directory)
            return
        except OSError as ex:
            if ex.errno == errno.EEXIST:
                raise

        time.sleep(0.1)

    # Final attempt, pass any exceptions up to caller.
    os.makedirs(directory)

def python_platform_identifier():
    system_from_python = platform.system().lower()
    if re.match('^cygwin_.*', system_from_python):
        return 'cygwin'
    return system_from_python

def run_compiler(options, ccinfo, default_return, flags=None):
    if flags is None:
        flags = []

    cc_bin = options.compiler_binary or ccinfo.binary_name

    cmd = cc_bin.split(' ') + flags

    try:
        logging.debug("Running '%s'", ' '.join(cmd))
        stdout, _ = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True).communicate()
        cc_output = stdout
    except OSError as ex:
        logging.warning('Could not execute %s: %s', cmd, ex)
        return default_return

    return cc_output

# This is for options that have --with-XYZ and --without-XYZ. If user does not
# set any of those, we choose a default here.
# Mutates `options`
def set_defaults_for_unset_options(options, info_arch, info_cc, info_os):
    if options.os is None:
        options.os = python_platform_identifier()
        logging.info('Guessing target OS is %s (use --os to set)', options.os)

    if options.os not in info_os:
        def find_canonical_os_name(os_name_variant):
            for (canonical_os_name, os_info) in info_os.items():
                if os_info.matches_name(os_name_variant):
                    return canonical_os_name
            return os_name_variant # not found
        options.os = find_canonical_os_name(options.os)

    def deduce_compiler_type_from_cc_bin(options):
        cc_bin = options.compiler_binary
        if cc_bin.find('clang') != -1 or cc_bin in ['emcc', 'em++']:
            return 'clang'
        if cc_bin.find('g++') != -1:
            return 'gcc'

        vers = run_compiler(options, None, '', ['--version'])
        if vers.find('clang') != -1:
            return 'clang'
        if vers.find('Free Software Foundation') != -1:
            return 'gcc'

        return None

    if options.compiler is None and options.compiler_binary is not None:
        options.compiler = deduce_compiler_type_from_cc_bin(options)

        if options.compiler is None:
            logging.error("Could not figure out what compiler type '%s' is, use --cc to set",
                          options.compiler_binary)

        if options.compiler == 'clang' and run_compiler(options, info_cc['clang'], '?', ['--version']).startswith('Apple clang'):
            options.compiler = 'xcode'

    if options.compiler is None and options.os in info_os:
        options.compiler = info_os[options.os].default_compiler

        if not have_program(info_cc[options.compiler].binary_name):
            logging.error("Default compiler is %s but could not find '%s'; use --cc to set",
                          options.compiler, info_cc[options.compiler].binary_name)

        logging.info('Guessing to use compiler %s (use --cc or CXX to set)', options.compiler)

    if options.cpu is None:
        (arch, cpu) = guess_processor(info_arch)
        options.arch = arch
        options.cpu = cpu
        logging.info('Guessing target processor is a %s (use --cpu to set)', options.arch)

    # OpenBSD uses an old binutils that does not support AVX2
    if options.os == 'openbsd':
        del info_cc['gcc'].isa_flags['avx2']

    if options.with_documentation is True:
        if options.with_sphinx is None and have_program('sphinx-build'):
            logging.info('Found sphinx-build (use --without-sphinx to disable)')
            options.with_sphinx = True
        if options.with_rst2man is None and have_program('rst2man'):
            logging.info('Found rst2man (use --without-rst2man to disable)')
            options.with_rst2man = True

    if options.with_pkg_config is None and options.os in info_os:
        options.with_pkg_config = info_os[options.os].uses_pkg_config

    if options.system_cert_bundle is None:
        default_paths = [
            '/etc/ssl/certs/ca-certificates.crt', # Ubuntu, Debian, Arch, Gentoo
            '/etc/pki/tls/certs/ca-bundle.crt', # RHEL
            '/etc/ssl/ca-bundle.pem', # SuSE
            '/etc/ssl/cert.pem', # OpenBSD, FreeBSD, Alpine
            '/etc/certs/ca-certificates.crt', # Solaris
            '/opt/local/share/curl/curl-ca-bundle.crt', # macOS with macPorts
        ]

        for path in default_paths:
            if os.access(path, os.R_OK):
                logging.info('Using %s as system certificate store', path)
                options.system_cert_bundle = path
                break
    else:
        if not os.access(options.system_cert_bundle, os.R_OK):
            logging.warning('System cert bundle "%s" not found, ignoring', options.system_cert_bundle)
            options.system_cert_bundle = None

# Mutates `options`
def canonicalize_options(options, info_os, info_arch):
    # canonical ARCH/CPU
    options.arch = canon_processor(info_arch, options.cpu)
    if options.arch is None:
        raise UserError('Unknown or unidentifiable processor "%s"' % (options.cpu))

    if options.cpu != options.arch:
        logging.info('Canonicalized CPU target %s to %s', options.cpu, options.arch)

    # select and sanity check build targets
    def canonicalize_build_targets(options):
        # --build-targets was not provided: build default targets
        if options.build_targets is None:
            return ["cli", "tests"]

        # flatten the list of multiple --build-targets="" and comma separation
        build_targets = [t.strip().lower() for ts in options.build_targets for t in ts.split(",")]

        # validate that all requested build targets are available
        for build_target in build_targets:
            if build_target not in ACCEPTABLE_BUILD_TARGETS:
                raise UserError("unknown build target: %s" % build_target)

        # building the shared lib desired and without contradiction?
        if options.build_shared_lib is None:
            options.build_shared_lib = "shared" in build_targets
        elif bool(options.build_shared_lib) != bool("shared" in build_targets):
            raise UserError("inconsistent usage of --enable/disable-shared-library and --build-targets")

        # building the static lib desired and without contradiction?
        if options.build_static_lib is None:
            options.build_static_lib = "static" in build_targets
        elif bool(options.build_static_lib) != bool("static" in build_targets):
            raise UserError("inconsistent usage of --enable/disable-static-library and --build-targets")

        return build_targets

    options.build_targets = canonicalize_build_targets(options)

    shared_libs_supported = options.os in info_os and info_os[options.os].building_shared_supported()

    if not shared_libs_supported:
        if options.build_shared_lib is True:
            logging.warning('Shared libs not supported on %s, disabling shared lib support', options.os)
            options.build_shared_lib = False
        elif options.build_shared_lib is None:
            logging.info('Shared libs not supported on %s, disabling shared lib support', options.os)

    if options.os == 'windows' and options.build_shared_lib is None and options.build_static_lib is None:
        options.build_shared_lib = True

    if options.with_stack_protector is None:
        if options.os in info_os:
            options.with_stack_protector = info_os[options.os].use_stack_protector

    if options.build_shared_lib is None:
        if options.os == 'windows' and options.build_static_lib:
            pass
        else:
            options.build_shared_lib = shared_libs_supported

    if options.build_static_lib is None:
        if options.os == 'windows' and options.build_shared_lib:
            pass
        else:
            options.build_static_lib = True

    if options.ldflags is not None:
        extra_libs = []
        link_to_lib = re.compile('^-l(.*)')
        for flag in options.ldflags.split(' '):
            match = link_to_lib.match(flag)
            if match:
                extra_libs.append(match.group(1))

        options.extra_libs += ','.join(extra_libs)

# Checks user options for consistency
# This method DOES NOT change options on behalf of the user but explains
# why the given configuration does not work.
def validate_options(options, info_os, info_cc, cc_version, available_module_policies):
    if options.name_amalgamation != 'botan_all':
        if options.name_amalgamation == '':
            raise UserError('Amalgamation basename must be non-empty')

        acceptable_name_re = re.compile('^[a-zA-Z0-9_]+$')
        if acceptable_name_re.match(options.name_amalgamation) is None:
            raise UserError("Amalgamation basename must match [a-zA-Z0-9_]+")

    if options.os == "java":
        raise UserError("Jython detected: need --os and --cpu to set target")

    if options.os not in info_os:
        raise UserError('Unknown OS "%s"; available options: %s' % (
            options.os, ' '.join(sorted(info_os.keys()))))

    if options.compiler not in info_cc:
        raise UserError('Unknown compiler "%s"; available options: %s' % (
            options.compiler, ' '.join(sorted(info_cc.keys()))))

    if options.cc_min_version is not None and not re.match(r'^[0-9]+\.[0-9]+$', options.cc_min_version):
        raise UserError("--cc-min-version must have the format MAJOR.MINOR")

    if options.module_policy and options.module_policy not in available_module_policies:
        raise UserError("Unknown module set %s" % options.module_policy)

    if options.cpu == 'llvm' or options.os == 'llvm':
        if options.compiler != 'clang':
            raise UserError('LLVM target requires using Clang')

        if options.cpu != 'llvm':
            raise UserError('LLVM target requires CPU target set to LLVM bitcode (llvm)')

        if options.os != 'llvm':
            raise UserError('Target OS is not an LLVM bitcode target')

    if options.cpu == 'wasm' or options.os == 'emscripten':
        if options.compiler != 'emcc':
            raise UserError('Emscripten target requires using emcc')

        if options.cpu != 'wasm':
            raise UserError('Emscripten target requires CPU target set to LLVM bitcode (wasm)')

        if options.os != 'emscripten':
            raise UserError('Target OS is not emscripten')

    if options.build_fuzzers is not None:
        if options.build_fuzzers not in ['libfuzzer', 'afl', 'klee', 'test']:
            raise UserError('Bad value to --build-fuzzers')

        if options.build_fuzzers == 'klee' and options.os != 'llvm':
            raise UserError('Building for KLEE requires targeting LLVM')

    if options.build_tool not in ['make', 'ninja']:
        raise UserError("Unknown --build-tool option (possible values: make, ninja)")

    if options.build_static_lib is False and options.build_shared_lib is False:
        raise UserError('With both --disable-static-library and --disable-shared-library, nothing to do')

    if options.os == 'windows' and options.build_static_lib is True and options.build_shared_lib is True:
        raise UserError('On Windows only one of static lib and DLL can be selected')

    if 'examples' in options.build_targets and 'boost' not in options.enabled_modules:
        raise UserError('Target examples requires --with-boost')

    if options.with_documentation is False:
        if options.with_doxygen:
            raise UserError('Using --with-doxygen plus --without-documentation makes no sense')
        if options.with_sphinx:
            raise UserError('Using --with-sphinx plus --without-documentation makes no sense')
        if options.with_pdf:
            raise UserError('Using --with-pdf plus --without-documentation makes no sense')

    if options.with_pdf and not options.with_sphinx:
        raise UserError('Option --with-pdf requires --with-sphinx')

    if options.ct_value_barrier_type:
        if options.ct_value_barrier_type not in ['asm', 'volatile', 'none']:
            raise UserError('Unknown setting "%s" for --ct-value-barrier-type' % (options.ct_value_barrier_type))

    if options.enable_stack_scrubbing and (options.compiler not in ['gcc'] or float(cc_version) < 14):
        raise UserError('Your compiler does not support stack scrubbing. Only GCC 14 and newer support this at the moment.')

    # Warnings
    if options.os == 'windows' and options.compiler != 'msvc':
        logging.warning('The windows target is oriented towards MSVC; maybe you want --os=cygwin or --os=mingw')

    if options.msvc_runtime:
        if options.compiler != 'msvc':
            raise UserError("Makes no sense to specify MSVC runtime for %s" % (options.compiler))

        if options.msvc_runtime not in ['MT', 'MD', 'MTd', 'MDd']:
            logging.warning("MSVC runtime option '%s' not known", options.msvc_runtime)

    if 'threads' in options.without_os_features:
        logging.warning('Disabling thread support will cause data races if threads are used by the application')

def run_compiler_preproc(options, ccinfo, source_file, default_return, extra_flags=None):
    if extra_flags is None:
        extra_flags = []

    cc_output = run_compiler(options, ccinfo, default_return, ccinfo.preproc_flags.split(' ') + extra_flags + [source_file])

    def cleanup_output(output):
        return ('\n'.join([l for l in output.splitlines() if l.startswith('#') is False])).strip()

    return cleanup_output(cc_output)

def calculate_cc_min_version(options, ccinfo, source_paths):
    version_patterns = {
        'msvc': r'^ *MSVC ([0-9]{2})([0-9]{2})$',
        'gcc': r'^ *GCC ([0-9]+) ([0-9]+)$',
        'clang': r'^ *CLANG ([0-9]+) ([0-9]+)$',
        'xcode': r'^ *XCODE ([0-9]+) ([0-9]+)$',
        'xlc': r'^ *XLC ([0-9]+) ([0-9]+)$',
        'emcc': r'^ *EMCC ([0-9]+) ([0-9]+)$',
    }

    unknown_pattern = r'UNKNOWN 0 0'

    cxx = ccinfo.basename

    if cxx not in version_patterns:
        logging.info("No compiler version detection available for %s", cxx)
        return "0.0"

    detect_version_source = os.path.join(source_paths.build_data_dir, "detect_version.cpp")

    cc_output = run_compiler_preproc(options, ccinfo, detect_version_source, "0.0")

    if re.search(unknown_pattern, cc_output) is not None:
        logging.warning('Failed to get version for %s from macro check', cxx)
        return "0.0"

    match = re.search(version_patterns[cxx], cc_output, flags=re.MULTILINE)
    if match is None:
        logging.warning("Tried to get %s version, but output '%s' is unexpected",
                        cxx, cc_output)
        return "0.0"

    major_version = int(match.group(1), 0)
    minor_version = int(match.group(2), 0)

    cc_version = "%d.%d" % (major_version, minor_version)

    if cc_version != '0.0':
        logging.info('Auto-detected compiler version %s %s', cxx, cc_version)

    if ccinfo.minimum_supported_version:
        # compare as floats
        min_ver = float(ccinfo.minimum_supported_version)
        our_ver = float(cc_version)

        if our_ver < min_ver:
            logging.error("This version of Botan requires at least %s %s",
                          cxx, ccinfo.minimum_supported_version)

    return cc_version

def check_compiler_arch(options, ccinfo, archinfo, source_paths):
    detect_version_source = os.path.join(source_paths.build_data_dir, 'detect_arch.cpp')

    abi_flags = ccinfo.mach_abi_link_flags(options).split(' ')
    cc_output = run_compiler_preproc(options, ccinfo, detect_version_source, 'UNKNOWN', abi_flags).lower()

    if cc_output == '':
        cc_output = run_compiler_preproc(options, ccinfo, detect_version_source, 'UNKNOWN').lower()

    if cc_output == 'unknown':
        logging.warning('Unable to detect target architecture via compiler macro checks')
        return None

    if cc_output not in archinfo:
        # Should not happen
        logging.warning("Error detecting compiler target arch: '%s'", cc_output)
        return None

    logging.info('Auto-detected compiler arch %s', cc_output)
    return cc_output

def do_io_for_build(cc, arch, osinfo, using_mods, info_modules, build_paths, source_paths, template_vars, options):
    try:
        robust_rmtree(build_paths.build_dir)
    except OSError as ex:
        if ex.errno != errno.ENOENT:
            logging.error('Problem while removing build dir: %s', ex)

    for build_dir in build_paths.build_dirs():
        try:
            robust_makedirs(build_dir)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                logging.error('Error while creating "%s": %s', build_dir, ex)

    def write_template_with_variables(sink, template, variables, postproc_fn = None):
        output = process_template(template, variables)
        if postproc_fn:
            output = postproc_fn(output)

        with open(sink, 'w', encoding='utf8') as f:
            f.write(output)

    def write_template(sink, template, postproc_fn = None):
        write_template_with_variables(sink, template, template_vars, postproc_fn)

    def in_build_dir(p):
        return os.path.join(build_paths.build_dir, p)
    def in_build_data(p):
        return os.path.join(source_paths.build_data_dir, p)
    def in_build_module_info(p):
        return os.path.join(build_paths.doc_module_info, p)

    write_template(in_build_dir('build.h'), in_build_data('buildh.in'))
    write_template(in_build_dir('target_info.h'), in_build_data('target_info.h.in'))
    write_template(in_build_dir('version_info.h'), in_build_data('version_info.h.in'))
    write_template(in_build_dir('botan.doxy'), in_build_data('botan.doxy.in'))

    if options.with_cmake_config:
        robust_makedirs(in_build_dir("cmake"))
        write_template(in_build_dir('cmake/botan-config.cmake'), in_build_data('botan-config.cmake.in'))
        write_template(in_build_dir('cmake/botan-config-version.cmake'), in_build_data('botan-config-version.cmake.in'))

    if 'botan_pkgconfig' in template_vars:
        write_template(template_vars['botan_pkgconfig'], in_build_data('botan.pc.in'))

    link_method = choose_link_method(options)

    def link_headers(headers, visibility, directory):
        logging.debug('Linking %d %s header files in %s', len(headers), visibility, directory)

        for header_file in headers:
            try:
                portable_symlink(header_file, directory, link_method)
            except OSError as ex:
                if ex.errno != errno.EEXIST:
                    raise UserError('Error linking %s into %s: %s' % (header_file, directory, ex)) from ex

    link_headers(build_paths.public_headers, 'public',
                 build_paths.public_include_dir)

    link_headers(build_paths.internal_headers, 'internal',
                 build_paths.internal_include_dir)

    link_headers(build_paths.external_headers, 'external',
                 build_paths.external_include_dir)

    if options.amalgamation:
        (amalg_cpp_files, amalg_headers) = AmalgamationGenerator(
            options.name_amalgamation, build_paths, using_mods, options).generate()
        build_paths.lib_sources = amalg_cpp_files
        template_vars['generated_files'] = ' '.join(amalg_cpp_files + amalg_headers)

        link_headers(amalg_headers, 'public', build_paths.public_include_dir)

        # Inserting an amalgamation generated using DLL visibility flags into a
        # binary project will either cause errors (on Windows) or unnecessary overhead.
        # Provide a hint
        if options.build_shared_lib:
            logging.warning('Unless you are building a DLL or .so from the amalgamation, use --disable-shared as well')

    template_vars.update(generate_build_info(build_paths, using_mods, cc, arch, osinfo, options))

    with open(os.path.join(build_paths.build_dir, 'build_config.json'), 'w', encoding='utf8') as f:
        json.dump(template_vars, f, sort_keys=True, indent=2)

    if options.with_compilation_database:
        write_template(in_build_dir('compile_commands.json'), in_build_data('compile_commands.json.in'))

    if options.build_tool == 'make':
        write_template(template_vars['makefile_path'], in_build_data('makefile.in'))
    elif options.build_tool == 'ninja':
        def escape_build_lines(contents):
            ninja_build_line = re.compile('^build (.*): (.*)')

            output = []
            for line in contents.split('\n'):
                match = ninja_build_line.match(line)
                if match:
                    escaped1 = match.group(1).replace(':', '$:')
                    escaped2 = match.group(2).replace(':', '$:')
                    output.append('build %s: %s' % (escaped1, escaped2))
                else:
                    output.append(line)

            return "\n".join(output)

        write_template(template_vars['ninja_build_path'], in_build_data('ninja.in'), escape_build_lines)

    if options.with_doxygen:
        for module_name, info in info_modules.items():
            dependencies = [dep for dep in info.requires if not dep.endswith('?dyn_load')]
            write_template_with_variables(in_build_module_info(module_name + '.dox'), in_build_data('module_info.in'),
                                          {
                                              'parent': info.parent_module,
                                              'identifier': module_name,
                                              'title': info.name,
                                              'internal': info.is_internal(),
                                              'virtual': info.is_virtual(),
                                              'deprecated': info.is_deprecated(),
                                              'experimental': info.is_experimental(),
                                              'brief': info.brief,
                                              'public_headers': info.header_public,
                                              'internal_headers': info.header_internal,
                                              'sources': info.sources(),
                                              'dependencies': dependencies,
                                              'os_features': info.os_features,
                                              'cpu_features': info.isa,
                                              'arch_requirements': info.arch,
                                              'compiler_requirements': info.cc
                                          })

    if options.with_rst2man:
        rst2man_file = os.path.join(build_paths.build_dir, 'botan.rst')
        cli_doc = os.path.join(source_paths.doc_dir, 'cli.rst')

        cli_doc_contents = open(cli_doc, encoding='utf8').readlines()

        while cli_doc_contents[0] != "\n":
            cli_doc_contents.pop(0)

        rst2man_header = """
botan
=============================

:Subtitle: Botan command line util
:Manual section: 1

        """.strip()

        with open(rst2man_file, 'w', encoding='utf8') as f:
            f.write(rst2man_header)
            f.write("\n")
            for line in cli_doc_contents:
                f.write(line)

    date = 'dated %d' % (Version.datestamp()) if Version.datestamp() != 0 else 'undated'

    logging.info('Botan %s (revision %s) (%s %s) build setup is complete',
                 Version.as_string(),
                 Version.vc_rev(),
                 Version.release_type(),
                 date)

    # Warn about build modes that are not safe for production:
    if options.unsafe_fuzzer_mode:
        logging.warning("Unsafe fuzzer mode is NOT SAFE FOR PRODUCTION")

    if options.unsafe_terminate_on_asserts:
        logging.warning("Terminating on assertion failures is NOT SAFE FOR PRODUCTION")

def list_os_features(all_os_features, info_os):
    for feat in all_os_features:
        os_with_feat = [o for o in info_os.keys() if feat in info_os[o].target_features]
        os_without_feat = [o for o in info_os.keys() if feat not in info_os[o].target_features]

        if len(os_with_feat) < len(os_without_feat):
            print("%s: %s" % (feat, ' '.join(sorted(os_with_feat))))
        else:
            print("%s: %s" % (feat, '!' + ' !'.join(sorted(os_without_feat))))
    return 0


def main(argv):
    """
    Main driver
    """

    options = process_command_line(argv[1:])

    setup_logging(options)

    source_paths = SourcePaths(os.path.dirname(argv[0]))

    info_modules = load_info_files(source_paths.lib_dir, 'Modules', "info.txt", ModuleInfo)

    if options.list_modules:
        public_modules = [name for (name, info) in info_modules.items() if info.is_public()]
        for mod in sorted(public_modules):
            print(mod)
        return 0

    info_arch = load_build_data_info_files(source_paths, 'CPU info', 'arch', ArchInfo)
    info_os = load_build_data_info_files(source_paths, 'OS info', 'os', OsInfo)
    info_cc = load_build_data_info_files(source_paths, 'compiler info', 'cc', CompilerInfo)
    info_module_policies = load_build_data_info_files(source_paths, 'module policy', 'policy', ModulePolicyInfo)

    all_os_features = sorted(set(flatten([o.target_features for o in info_os.values()])))
    all_defined_isas = set(flatten([a.isa_extensions for a in info_arch.values()]))

    if options.list_os_features:
        return list_os_features(all_os_features, info_os)

    for mod in info_modules.values():
        mod.cross_check(info_arch, info_cc, all_os_features, all_defined_isas)

    for cc in info_cc.values():
        cc.cross_check(info_os, info_arch, all_defined_isas)

    for policy in info_module_policies.values():
        policy.cross_check(info_modules)

    logging.info('%s invoked with options "%s"', argv[0], ' '.join(argv[1:]))
    logging.info('Configuring to build Botan %s (revision %s)',
                 Version.as_string(), Version.vc_rev())
    logging.info('Python version: "%s"', sys.version.replace('\n', ''))

    take_options_from_env(options)

    logging.info('Autodetected platform information: OS="%s" machine="%s" proc="%s"',
                 platform.system(), platform.machine(), platform.processor())

    cpu_names = sorted(flatten([[ainfo.basename] + ainfo.aliases for ainfo in info_arch.values()]))
    logging.debug('Known CPU names: %s', ' '.join(cpu_names))

    set_defaults_for_unset_options(options, info_arch, info_cc, info_os)
    canonicalize_options(options, info_os, info_arch)

    cc = info_cc[options.compiler]
    arch = info_arch[options.arch]
    osinfo = info_os[options.os]
    module_policy = info_module_policies[options.module_policy] if options.module_policy else None

    if options.enable_cc_tests:
        cc_min_version = options.cc_min_version or calculate_cc_min_version(options, cc, source_paths)

        if options.arch not in ['generic', 'llvm']:
            cc_arch = check_compiler_arch(options, cc, info_arch, source_paths)
            if cc_arch is not None and cc_arch != options.arch:
                logging.error("Configured target is %s but compiler probe indicates %s", options.arch, cc_arch)
    else:
        cc_min_version = options.cc_min_version or "0.0"

    validate_options(options, info_os, info_cc, cc_min_version, info_module_policies)

    logging.info('Target is %s:%s-%s-%s',
                 options.compiler, cc_min_version, options.os, options.arch)

    chooser = ModulesChooser(info_modules, module_policy, arch, osinfo, cc, cc_min_version, options)
    loaded_module_names = chooser.choose()
    using_mods = [info_modules[modname] for modname in loaded_module_names]
    not_using_mods = [modinfo for modname, modinfo in info_modules.items() if modname not in loaded_module_names]

    build_paths = BuildPaths(source_paths, options, using_mods)
    build_paths.public_headers.append(os.path.join(build_paths.build_dir, 'build.h'))
    for internal_headers in ['target_info.h', 'version_info.h']:
        build_paths.internal_headers.append(os.path.join(build_paths.build_dir, internal_headers))

    template_vars = create_template_vars(source_paths, build_paths, options, using_mods, not_using_mods, cc, arch, osinfo)

    # Now we start writing to disk
    do_io_for_build(cc, arch, osinfo, using_mods, info_modules, build_paths, source_paths, template_vars, options)

    return 0

if __name__ == '__main__':
    try:
        sys.exit(main(argv=sys.argv))
    except UserError as e:
        logging.debug(traceback.format_exc())
        logging.error(e)
    except Exception as e: # pylint: disable=broad-except
        # error() will stop script, so wrap all information into one call
        logging.error("""%s
An internal error occurred.

Don't panic, this is probably not your fault! Please open an issue
with the entire output at https://github.com/randombit/botan

You'll meet friendly people happy to help!""", traceback.format_exc())

    sys.exit(0)
