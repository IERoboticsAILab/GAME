# -*- coding: utf-8 -*-
"""
Minimal pure-Python OptiTrack NatNet client.

Connects to OptiTrack Motive over the NatNet protocol (no external/native
dependencies -- only the Python standard library) and reports rigid-body poses.
It negotiates the NatNet version from Motive's ServerInfo, then parses the
rigid bodies out of each frame of data.

Rigid bodies are matched by their Motive "Streaming ID" (Properties -> Streaming
ID in Motive), which is the most robust key. Names are resolved best-effort from
the model definitions when available.

Supports NatNet up to 4.5 by requesting a compatible bitstream version from the
server when running in unicast mode (Motive 3.0+).

Usage:
    client = NatNetClient(server_ip='192.168.1.100')   # multicast (Motive default)
    client.on_rigid_body = lambda rb: print(rb.id, rb.position, rb.rotation)
    client.start()
    ...
    client.close()
"""
import select
import socket
import struct
import threading
import traceback
import time
from collections import namedtuple

# NatNet message IDs
NAT_CONNECT = 0
NAT_SERVERINFO = 1
NAT_REQUEST = 2
NAT_RESPONSE = 3
NAT_REQUEST_MODELDEF = 4
NAT_MODELDEF = 5
NAT_REQUEST_FRAMEOFDATA = 6
NAT_FRAMEOFDATA = 7
NAT_MESSAGESTRING = 8
NAT_DISCONNECT = 9
NAT_KEEPALIVE = 10

# Motive drops a unicast client that goes quiet; keep well under that.
KEEPALIVE_PERIOD = 2.0

# The bitstream version this parser is known to handle correctly.
# When a newer server is detected on unicast, we request this version.
SUPPORTED_BITSTREAM = (4, 1, 0, 0)

Quaternion = namedtuple('Quaternion', ['x', 'y', 'z', 'w'])

RigidBody = namedtuple('RigidBody', ['id', 'name', 'position', 'rotation', 'tracking_valid'])


class _ParseError(Exception):
    """Raised when the cursor goes out of bounds or data is corrupt."""
    pass


class _Cursor:
    """Sequential little-endian reader over a bytes buffer with bounds checking."""

    def __init__(self, data, offset=0):
        self.data = data
        self.offset = offset

    def _check(self, n):
        if self.offset + n > len(self.data):
            raise _ParseError(
                "read %d bytes at offset %d overflows buffer of %d bytes"
                % (n, self.offset, len(self.data))
            )

    def int32(self):
        self._check(4)
        v = struct.unpack_from('<i', self.data, self.offset)[0]
        self.offset += 4
        return v

    def uint32(self):
        self._check(4)
        v = struct.unpack_from('<I', self.data, self.offset)[0]
        self.offset += 4
        return v

    def uint16(self):
        self._check(2)
        v = struct.unpack_from('<H', self.data, self.offset)[0]
        self.offset += 2
        return v

    def int16(self):
        self._check(2)
        v = struct.unpack_from('<h', self.data, self.offset)[0]
        self.offset += 2
        return v

    def float(self):
        self._check(4)
        v = struct.unpack_from('<f', self.data, self.offset)[0]
        self.offset += 4
        return v

    def floats(self, n):
        self._check(4 * n)
        v = struct.unpack_from('<%df' % n, self.data, self.offset)
        self.offset += 4 * n
        return v

    def skip(self, n_bytes):
        if n_bytes < 0 or self.offset + n_bytes > len(self.data):
            raise _ParseError(
                "skip(%d) out of bounds at offset %d (buffer %d)"
                % (n_bytes, self.offset, len(self.data))
            )
        self.offset += n_bytes

    def string(self):
        try:
            end = self.data.index(b'\0', self.offset)
        except ValueError:
            raise _ParseError(
                "unterminated string at offset %d (buffer %d)"
                % (self.offset, len(self.data))
            )
        s = self.data[self.offset:end].decode('utf-8', errors='replace')
        self.offset = end + 1
        return s


class NatNetClient(threading.Thread):
    def __init__(self, server_ip, local_ip='0.0.0.0',
                 multicast_address='239.255.42.99',
                 command_port=1510, data_port=1511, use_multicast=True):
        super().__init__(daemon=True)
        self.server_ip = server_ip
        self.local_ip = local_ip
        self.multicast_address = multicast_address
        self.command_port = command_port
        self.data_port = data_port
        self.use_multicast = use_multicast

        # (major, minor, build, revision); assume 3.0 until ServerInfo arrives.
        self.nat_net_version = (3, 0, 0, 0)
        self.server_info_received = threading.Event()

        # Callback invoked for every rigid body in every frame.
        self.on_rigid_body = None
        # id -> name, populated best-effort from model definitions.
        self.id_to_name = {}
        # Callback/parse failures, kept instead of killing the receive thread.
        self.errors = 0
        self.last_error = None

        self._stay_open = True
        self._data_sock = None
        self._command_sock = None
        self._bitstream_requested = False
        self._last_frame_sized = None

    # --- public API --------------------------------------------------------
    def close(self):
        self._stay_open = False

    def run(self):
        self._command_sock = self._make_command_socket()
        self._data_sock = self._make_data_socket()

        # Ask the server who it is (gets us the NatNet version) and for the
        # model definitions (rigid-body names).
        self._send_command(NAT_CONNECT)
        self._send_command(NAT_REQUEST_MODELDEF)

        self._command_sock.setblocking(False)
        self._data_sock.setblocking(False)

        next_def_request = time.monotonic() + 0.5
        next_keepalive = time.monotonic() + KEEPALIVE_PERIOD
        socks = [self._data_sock, self._command_sock]
        while self._stay_open:
            now = time.monotonic()
            # Motive stops streaming to a silent unicast client after ~10 s.
            # Without this the pose feed dies mid-flight and the drone keeps
            # flying on a frozen estimate.
            if now > next_keepalive:
                self._send_command(NAT_KEEPALIVE)
                next_keepalive = now + KEEPALIVE_PERIOD
            # Motive can miss the first model-def request; retry until names land.
            if not self.id_to_name and now > next_def_request:
                self._send_command(NAT_REQUEST_MODELDEF)
                next_def_request = now + 0.5
            # Drain everything that is ready: at 100+ Hz, reading one packet per
            # pass falls behind and the pose we act on goes stale.
            ready, _, _ = select.select(socks, [], [], 0.1)
            for sock in ready:
                while self._poll(sock):
                    pass

        for s in (self._data_sock, self._command_sock):
            try:
                s.close()
            except Exception:
                pass

    # --- sockets -----------------------------------------------------------
    def _make_data_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.use_multicast:
            s.bind(('', self.data_port))
            mreq = struct.pack('4sl', socket.inet_aton(self.multicast_address),
                               socket.INADDR_ANY)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        else:
            s.bind((self.local_ip, self.data_port))
        return s

    def _make_command_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.bind(('', 0))
        return s

    def _send_command(self, command):
        # Body-less commands (connect / request model def / request frame).
        packet = struct.pack('<HH', command, 0)
        self._command_sock.sendto(packet, (self.server_ip, self.command_port))

    def _send_string_command(self, command_str):
        """Send a NAT_REQUEST with a string payload (e.g. remote commands)."""
        encoded = command_str.encode('utf-8') + b'\0'
        packet = struct.pack('<HH', NAT_REQUEST, len(encoded)) + encoded
        self._command_sock.sendto(packet, (self.server_ip, self.command_port))

    def _request_bitstream_version(self, major, minor, build=0, rev=0):
        """Ask Motive to stream a specific NatNet bitstream version.

        This only works in unicast mode and requires Motive >= 3.0 / NatNet >= 4.0.
        Motive will acknowledge by sending updated data in the requested format.
        """
        cmd = "Bitstream,%d.%d.%d.%d" % (major, minor, build, rev)
        self._send_string_command(cmd)
        print("NatNet: requested bitstream version %d.%d.%d.%d" % (major, minor, build, rev))

    # --- receive loop ------------------------------------------------------
    def _poll(self, sock):
        try:
            data, _ = sock.recvfrom(65535)
        except (socket.timeout, BlockingIOError):
            return False
        except OSError:
            return False
        if len(data) >= 4:
            try:
                self._dispatch(data)
            except _ParseError as e:
                # Parser went out of bounds -- likely a bitstream version
                # mismatch that the version-request hasn't resolved yet.
                # Keep the stream alive; do not print every frame.
                self.errors += 1
                if self.errors <= 3:
                    print('NatNet parse error (stream kept alive): %s' % e)
            except Exception as e:
                # A raising callback used to kill this thread outright: the pose
                # stream stopped for good while the drone kept flying on a
                # frozen estimate. Keep receiving and report instead.
                self.errors += 1
                if self.last_error is None:
                    self.last_error = traceback.format_exc()
                    print('NatNet dispatch error (stream kept alive): %r' % e)
        return True

    def _dispatch(self, data):
        message_id = struct.unpack_from('<H', data, 0)[0]
        # bytes 2:4 are packet size; payload starts at offset 4
        if message_id == NAT_FRAMEOFDATA:
            self._unpack_frame(data, 4)
        elif message_id == NAT_SERVERINFO:
            self._unpack_server_info(data, 4)
        elif message_id == NAT_MODELDEF:
            try:
                self._unpack_model_def(data, 4)
            except Exception:
                # Names are best-effort; matching by ID still works.
                pass

    # --- parsers -----------------------------------------------------------
    def _unpack_server_info(self, data, offset):
        # application name occupies 256 bytes, then Version[4], NatNetVersion[4]
        natnet = data[offset + 256 + 4: offset + 256 + 8]
        if len(natnet) == 4:
            server_version = tuple(natnet)
            print("NatNet: server reports bitstream version %d.%d.%d.%d"
                  % server_version)

            # If the server's bitstream is newer than what this parser handles,
            # request the known-good version.  This only works on unicast
            # (multicast sends one stream for all clients).
            if (not self.use_multicast
                    and server_version > SUPPORTED_BITSTREAM
                    and not self._bitstream_requested):
                self._request_bitstream_version(*SUPPORTED_BITSTREAM)
                self._bitstream_requested = True
                # Tell the parser to expect the version we requested, not the
                # server's native one.
                self.nat_net_version = SUPPORTED_BITSTREAM
                # Re-request model definitions so names arrive in the
                # requested bitstream format.
                self._send_command(NAT_REQUEST_MODELDEF)
            else:
                self.nat_net_version = server_version

        self.server_info_received.set()

    def _frame_is_sized(self, data, offset):
        """Does this frame put an int32 byte-count after each section count?

        Bitstream 4.1 does; 4.5 does not, it goes straight into the first
        marker set's name. We cannot rely on the negotiated version to decide,
        because Motive keeps sending frames in the OLD format for a moment
        after acknowledging a bitstream downgrade -- during that window the
        version says 4.1 while the bytes are still 4.5, and reading a model
        name as a length gives the classic skip(1919306049).

        So look at the bytes: a section length has its top bytes zero, while a
        model name starts with printable ASCII. Marker sets in Motive are
        always named, so this is unambiguous in practice.
        """
        probe = data[offset + 8: offset + 12]      # after frame number + count
        if len(probe) < 4:
            return False
        if probe[0] == 0:                          # empty string, or tiny size
            return True
        printable = 0x20 <= probe[0] <= 0x7E
        looks_like_size = probe[3] == 0 and probe[2] < 0x10
        if printable and not looks_like_size:
            return False
        if looks_like_size and not printable:
            return True
        # Both plausible: trust a size only if it lands inside the packet.
        size = int.from_bytes(probe, 'little')
        return 0 < size <= len(data) - (offset + 12)

    def _unpack_frame(self, data, offset):
        c = _Cursor(data, offset)
        major = self.nat_net_version[0]
        # Detected per frame rather than taken from the negotiated version --
        # see _frame_is_sized().
        sized = self._frame_is_sized(data, offset)
        if sized != self._last_frame_sized:
            print("NatNet: frame layout is %s"
                  % ("size-prefixed (bitstream 4.1)" if sized
                     else "not size-prefixed (bitstream 4.5)"))
            self._last_frame_sized = sized

        c.int32()  # frame number

        # Marker sets
        n = c.int32()
        if sized:
            c.skip(c.int32())          # jump the whole section
        else:
            for _ in range(n):
                c.string()             # model name
                c.skip(12 * c.int32())

        # Legacy "other"/unlabeled markers
        n_other = c.int32()
        if sized:
            c.skip(c.int32())
        else:
            c.skip(12 * n_other)

        # Rigid bodies
        n_rb = c.int32()
        if sized:
            c.int32()                  # section size (we parse this one)
        for _ in range(n_rb):
            rb_id = c.int32()
            x, y, z = c.floats(3)
            qx, qy, qz, qw = c.floats(4)

            if major < 3:
                # markers are embedded in the rigid body pre-NatNet 3.0
                m = c.int32()
                c.skip(12 * m)                 # marker positions
                if self.nat_net_version >= (2, 0, 0, 0):
                    c.skip(4 * m)              # marker ids
                    c.skip(4 * m)              # marker sizes

            if self.nat_net_version >= (2, 0, 0, 0):
                c.float()                      # mean marker error

            tracking_valid = True
            if self.nat_net_version >= (2, 6, 0, 0) or major >= 3:
                params = c.int16()
                tracking_valid = bool(params & 0x01)

            if self.on_rigid_body:
                rb = RigidBody(
                    id=rb_id,
                    name=self.id_to_name.get(rb_id),
                    position=(x, y, z),
                    rotation=Quaternion(qx, qy, qz, qw),
                    tracking_valid=tracking_valid,
                )
                self.on_rigid_body(rb)
            # Remaining frame sections (skeletons, labeled markers, devices,
            # timing) are not needed and are ignored.

    def _unpack_model_def(self, data, offset):
        c = _Cursor(data, offset)
        major = self.nat_net_version[0]
        dataset_count = c.int32()
        for _ in range(dataset_count):
            dtype = c.int32()
            end = None
            if major >= 3:
                end = c.offset + 4 + c.int32()   # per-dataset byte count
            if dtype == 1:                       # rigid body
                self._read_rigid_body_def(c, major)
            elif end is None:
                if dtype == 0:                   # marker set
                    c.string()
                    for _ in range(c.int32()):
                        c.string()
                elif dtype == 2:                 # skeleton
                    c.string()
                    c.int32()                    # skeleton id
                    for _ in range(c.int32()):
                        self._read_rigid_body_def(c, major)
                else:
                    break                        # can't skip safely, stop
            if end is not None:
                c.offset = end                   # jump past whatever it was

    def _read_rigid_body_def(self, c, major):
        name = c.string() if self.nat_net_version >= (2, 0, 0, 0) else ''
        rb_id = c.int32()
        c.int32()                               # parent id
        c.floats(3)                             # offset x,y,z
        if major >= 3:
            m = c.int32()
            c.skip(12 * m)                      # marker positions
            c.skip(4 * m)                       # active marker labels
            if major >= 4:
                for _ in range(m):
                    c.string()                  # marker names
        if name:
            self.id_to_name[rb_id] = name
