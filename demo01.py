import PIL.Image
import torch.nn
import random
from io import BytesIO
import torch
import torch.nn.functional as F
import pywebio
from models.ecl import ECL_model, balanced_proxies
from torchvision import transforms
import time
import numpy as np

from pyecharts.charts import Bar, Page
from pyecharts import options as opts


def bar_base(data) -> Bar:
    c = (
        Bar()
        # Bar({"theme": ThemeType.MACARONS})
        # .add_xaxis(["Melanoma", "Nevus", "BCC", "Bowen's disease", "BK", "Dermatofibroma", "Vascular"])
        .add_xaxis(["血管损伤", "鲍恩病", "纤维瘤", "黑色素痣", "基底细胞癌", "良性角化病", "黑色素瘤"])
        .add_yaxis("output_value", data, markpoint_opts=["max"])
        .set_global_opts(
            title_opts={"text": "模型输出", "subtext": ""},)
        )
    return c


def refresh():
    pywebio.output.clear()
    page1()

'''function for getting proxies number'''
def get_proxies_num(cls_num_list):
    ratios = [max(np.array(cls_num_list)) / num for num in cls_num_list]
    prototype_num_list = []
    for ratio in ratios:
        if ratio == 1:
            prototype_num = 1
        else:
            prototype_num = int(ratio // 10) + 2
        prototype_num_list.append(prototype_num)
    assert len(prototype_num_list) == len(cls_num_list)
    return prototype_num_list


def generate_random_str(target_length=32):
    random_str = ''
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(target_length):
        random_str += base_str[random.randint(0, length)]
    return random_str


def compare_ans(a, ans):
    if a == ans:
        return True
    else:
        return False





def page1(is_demo=True):
    user_ip = str(pywebio.session.info.user_ip)+generate_random_str(16)
    ans = "正常"
    
    ans_list = ["血管损伤", "鲍恩病", "纤维瘤", "黑色素痣", "基底细胞癌", "良性角化病", "黑色素瘤"]
    ans_y = [1, 0, 0, 0, 0, 0, 0, 0]
    chart_html = bar_base(ans_y).render_notebook()
    temp_file_path = "demo.jpg"
    graph_img = PIL.Image.open("./data/net_graph.png")
    train_img = PIL.Image.open("./data/train_process2.png")
    skin_ori = PIL.Image.open("./data/1.png")
    skin_gtmask = PIL.Image.open("./data/2.png")
    skin_predmask = PIL.Image.open("./data/3.png")


    while 1:
        try:
            pywebio.output.put_warning("识别结果仅供参考", closable=True, position=- 1)

            pywebio.output.put_html("<h1><center>SkinWatch 皮肤癌智能辅助诊断工具</center></h1><hr>")
            # .add_xaxis(["黑色素瘤", "黑色素痣", "基底细胞癌", "鲍恩病", "角化病", "纤维瘤", "血管损伤"])
            no_content = [pywebio.output.put_markdown("正常")]
            va_content = [pywebio.output.put_markdown("血管损伤")]
            me_content = [pywebio.output.put_markdown("黑色素瘤")]
            ne_content = [pywebio.output.put_markdown("黑色素痣")]
            bc_content = [pywebio.output.put_markdown("基底细胞癌")]
            bd_content = [pywebio.output.put_markdown("鲍恩病")]
            bk_content = [pywebio.output.put_markdown("角化病")]
            df_content = [pywebio.output.put_markdown("纤维瘤")]
            

            pywebio.output.put_row(
                [pywebio.output.put_scope(name="chart", content=[pywebio.output.put_html(chart_html)])
                 ],
            )

            pywebio.output.put_row(
                # ["血管损伤", "鲍恩病", "纤维瘤", "黑色素痣", "基底细胞癌", "良性角化病", "黑色素瘤"]
                # .add_xaxis(["Melanoma", "Nevus", "BCC", "Bowen's disease", "BK", "Dermatofibroma", "Vascular"])
                [pywebio.output.put_collapse("Normal", no_content, open=compare_ans("正常", ans)),
                 pywebio.output.put_collapse("VASC", va_content, open=compare_ans("血管损伤", ans)),
                 pywebio.output.put_collapse("AKIEC", bd_content, open=compare_ans("鲍恩病", ans)),
                 pywebio.output.put_collapse("DF", df_content, open=compare_ans("纤维瘤", ans)),
                 pywebio.output.put_collapse("NV", me_content, open=compare_ans("黑色素痣", ans)),
                 pywebio.output.put_collapse("BCC", bc_content, open=compare_ans("基底细胞癌", ans)),
                 pywebio.output.put_collapse("BKL", bk_content, open=compare_ans("良性角化病", ans)),
                 pywebio.output.put_collapse("MEL", ne_content, open=compare_ans("黑色素瘤", ans))],
            )

            more_content = [
                pywebio.output.put_markdown("ref: [点击进入分割页面](https://github.com/moboehle/Pytorch-LRP)"),
                pywebio.output.put_table(
                [
                    [pywebio.output.put_image(skin_ori),
                    pywebio.output.put_image(skin_gtmask),
                    pywebio.output.put_image(skin_predmask),],
                ]
                )
            ]
            f = open("models\ecl.py", "r", encoding="UTF-8")
            code = f.read()
            f.close()
            
            pywebio.output.put_collapse("分割图demo", more_content, open=True, position=- 1)
            pywebio.output.put_row([
                pywebio.output.put_collapse("模型信息", [pywebio.output.put_image(graph_img)], open=True, position=- 1),
                pywebio.output.put_collapse("训练信息", [pywebio.output.put_image(train_img),], open=True, position=- 1)
            ])
            pywebio.output.put_collapse("模型代码", [pywebio.output.put_code(code, "python")], open=False, position=- 1)
            # pywebio.output.put_markdown("ref: [代码仓库](https://github.com/LittleDuckH")
            pywebio.output.put_markdown("datasets: [数据集](https://adni.loni.usc.edu)")

            action = pywebio.input.actions(' ',
                                           [{'label': "上传.jpg图像", 'value': "上传.jpg图像", 'color': 'warning'},
                                            {'label': "使用demo.jpg", 'value': "使用demo.jpg", 'color': 'info'}
                                            ])
            if action == "使用demo.jpg":
                is_demo = True
            if action == "上传.jpg图像":
                is_demo = False
            
            ###################################################################################

            if is_demo is False:
                try:
                    inpic = pywebio.input.file_upload(label="上传皮肤图片(.jpg)")
                    inpic = BytesIO(inpic['content'])
                    temp_file_path = "./temp_data/" + generate_random_str() + ".jpg"
                    with open(temp_file_path, 'wb') as file:
                        file.write(inpic.getvalue())  # 保存到本地
                except:
                    pywebio.output.toast("输入错误，请上传jpg格式图片", color="warn")
                    refresh()

            if is_demo is True:
                is_demo = False
                temp_file_path = "demo.jpg"
            try:
                pywebio.output.popup("AI识别中", [pywebio.output.put_row(
                    [pywebio.output.put_loading(shape="grow", color="success")],
                )])
                print("AI识别中...")
            except:
                pywebio.output.toast("识别错误", color="warn")


            ##############################################################################
            print(1)
            try:
                '''load model'''
                model = ECL_model(num_classes=7, feat_dim=128)
                print(2)
                proxy_num_list = get_proxies_num([84, 195, 69, 4023, 308, 659, 667])
                print(3)
                model_proxy = balanced_proxies(dim=128, proxy_num=sum(proxy_num_list))
                print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
                print("Model_proxy size: {:.5f}M".format(sum(p.numel() for p in model_proxy.parameters())/1000000.0))
                print("=============model init done=============")
            except Exception:
                pywebio.output.toast("输入处理错误1", color="warn")
                refresh()

            print(111)
            img = None
            try:
                model.load_state_dict(torch.load(r"data\model_save\bestacc_model_106.pth", map_location=torch.device('cpu')), strict=True)
                # print(1)
                model.eval()
                # print(2)
                img = PIL.Image.open(temp_file_path)
                # print(3)
                with torch.no_grad():
                    # Load and preprocess the image
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    img = transform(img).unsqueeze(0)

                    # Perform inference
                    output = model(img)
                    # ans = torch.argmax(output, dim=1)
                    print(output)
                    predicted_results = torch.argmax(output, dim=1)
                    
                    # Output results
                    print("Predicted class:", predicted_results.item())
                    print("Predicted class:", predicted_results)

                    # 应用softmax函数
                    softmax_output = F.softmax(output, dim=1)
                    # 保留两位小数并转换为字符串格式
                    softmax_output_str = ["{:.4f}".format(prob.item()) for prob in softmax_output[0]]
                    print("Softmax output:", softmax_output_str)

                    for i in range(7):
                        ans_y[i] = softmax_output_str[i]
                    # print(max(ans_y))
                    max = 0.0
                    for i in ans_y:
                        if float(i) > max:
                            max = float(i)
                    if max < 0.5:
                        ans = '正常'
                    else:
                        ans = ans_list[predicted_results]

            except Exception:
                pywebio.output.toast("输入处理错误2", color="warn")
                refresh()
            # print(4)

            chart_html = bar_base([ans_y[0], ans_y[1], ans_y[2], ans_y[3], ans_y[4], ans_y[5], ans_y[6], ans_y[7]]).render_notebook()
            with pywebio.output.use_scope(name="chart") as scope_name:
                pywebio.output.clear()
                pywebio.output.put_html(chart_html)
            # print(chart_html)
            # print(5)
            show_result = [pywebio.output.put_markdown("诊断为：\n # " + ans)]
            pywebio.output.popup(title='AI识别结果', content=show_result)
            # print(6)
            pywebio.output.clear()

        except Exception:
            continue


if __name__ == "__main__":
    # page1()
    pywebio.platform.start_server(
        # applications=[page1, ],
        page1,
        debug=False,
        auto_open_webbrowser=False,
        remote_access=False,
        cdn=False,
        port=6006
    )

