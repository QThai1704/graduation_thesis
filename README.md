# Hướng dẫn Tải về và Lưu trữ Các File Trọng Số

Các file trọng số cần thiết cho dự án này không được đẩy lên GitHub vì lý do bảo mật hoặc kích thước. Bạn cần tải về các file này từ các liên kết bên dưới và lưu trữ chúng vào thư mục `weights/` trong dự án của bạn.

## Danh sách các File Trọng Số và Liên Kết Tải về

1. **File Trọng Số 1:**
   - **Tên:** `weights1.pth`
   - **Đường dẫn tải về:** [Tải về weights1.pth](https://example.com/weights1.pth)
   - **Thư mục lưu trữ:** `weights/`

2. **File Trọng Số 2:**
   - **Tên:** `weights2.pth`
   - **Đường dẫn tải về:** [Tải về weights2.pth](https://example.com/weights2.pth)
   - **Thư mục lưu trữ:** `weights/`

## Hướng Dẫn Lưu Trữ Các File Trọng Số

1. **Tải về các file trọng số:**
   - Truy cập các liên kết tải về ở trên và tải về các file trọng số cần thiết.

2. **Tạo thư mục `weights` nếu chưa tồn tại:**
   - Mở terminal và điều hướng đến thư mục gốc của dự án.
   - Chạy lệnh sau để tạo thư mục `weights`:
     ```bash
     mkdir -p weights
     ```

3. **Di chuyển các file trọng số vào thư mục `weights`:**
   - Sử dụng các lệnh sau để di chuyển các file đã tải về vào thư mục `weights`:
     ```bash
     mv /path/to/downloaded/weights1.pth weights/
     mv /path/to/downloaded/weights2.pth weights/
     ```
