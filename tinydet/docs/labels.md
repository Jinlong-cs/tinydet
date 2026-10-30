# ToC detection labels

[`../labels.json`](../labels.json) defines the 55 ToC classes. The array index is the YOLO class ID
(0–54); there is no background class. Keep this order with each dataset, checkpoint and exported model.
These are custom IDs, not COCO IDs. COCO weights remain a training initializer; changing this file does not
relabel existing annotations or turn a COCO checkpoint into a ToC model.

## Classes

IDs 0–47 cover objects and entrances. IDs 48–54 provide region proposals; their boxes do not supply exact
boundaries, thickness or depth. All classes use normalized YOLO boxes: `class_id cx cy width height`.

| ID  | Label                   | Chinese name     |
| --- | ----------------------- | ---------------- |
| 0   | `table`                 | 桌子／茶几       |
| 1   | `chair`                 | 椅子             |
| 2   | `cabinet`               | 柜子             |
| 3   | `storage_box`           | 收纳盒           |
| 4   | `couch`                 | 沙发             |
| 5   | `houseplant`            | 室内绿植         |
| 6   | `kitchen_island`        | 岛台             |
| 7   | `bed`                   | 床               |
| 8   | `refrigerator`          | 冰箱             |
| 9   | `dishwasher`            | 洗碗机           |
| 10  | `oven`                  | 烤箱             |
| 11  | `microwave`             | 微波炉           |
| 12  | `dish_sterilizer`       | 消毒柜           |
| 13  | `cooktop`               | 灶具／炉面       |
| 14  | `laundry_machine`       | 洗烘设备         |
| 15  | `cleaning_robot`        | 清洁机器人       |
| 16  | `upright_floor_cleaner` | 立式／手持洗地机 |
| 17  | `air_purifier`          | 空气净化器       |
| 18  | `tv`                    | 电视             |
| 19  | `speaker`               | 音箱             |
| 20  | `charging_dock`         | 充电基站         |
| 21  | `cup`                   | 杯子             |
| 22  | `bottle`                | 瓶子             |
| 23  | `tissue_box`            | 纸巾盒           |
| 24  | `bowl`                  | 碗／食盆         |
| 25  | `trash_bin`             | 垃圾桶           |
| 26  | `bathroom_scale`        | 体重秤           |
| 27  | `soft_bag`              | 软袋             |
| 28  | `suitcase`              | 行李箱           |
| 29  | `service_cart`          | 服务推车         |
| 30  | `footwear`              | 鞋／拖鞋         |
| 31  | `loose_textile`         | 散放织物         |
| 32  | `paper_litter`          | 散落纸品         |
| 33  | `fruit`                 | 水果             |
| 34  | `snack`                 | 零食             |
| 35  | `toy`                   | 玩具             |
| 36  | `office_supplies`       | 文具             |
| 37  | `knife`                 | 刀具             |
| 38  | `person`                | 人               |
| 39  | `dog`                   | 狗               |
| 40  | `cat`                   | 猫               |
| 41  | `bird`                  | 鸟               |
| 42  | `cable`                 | 线缆             |
| 43  | `pet_feces`             | 宠物粪便         |
| 44  | `door_leaf`             | 普通门扇         |
| 45  | `doorway`               | 普通门洞／入口   |
| 46  | `elevator_entrance`     | 电梯入口         |
| 47  | `sign_panel`            | 标牌             |
| 48  | `rug`                   | 地毯／地垫       |
| 49  | `mirror`                | 镜面             |
| 50  | `glass_surface`         | 透明玻璃面       |
| 51  | `liquid_spill`          | 液体／湿污渍区域 |
| 52  | `dried_stain`           | 干结污渍区域     |
| 53  | `granular_debris`       | 散落颗粒区域     |
| 54  | `hair_clump`            | 毛发聚集／毛团   |

## Annotation rules

- Merge tables/coffee tables, ready-made/custom cabinets, washers/dryers, plastic/cloth bags, and loose
  garments/rags/towels. Keep useful subtypes in source metadata; the detector does not predict those attributes.
- `loose_textile` excludes worn clothing, normal bedding, curtains and rugs. `bowl` includes ordinary and pet bowls.
  `cleaning_robot` covers autonomous cleaners; `upright_floor_cleaner` covers upright/handheld floor cleaners.
- Label each entity once: recognizable snack packages are `snack`, while bags with unknown contents remain `soft_bag`.
  `knife` takes priority over `office_supplies`, loose sheets use `paper_litter`, and tissue dispensers use `tissue_box`.
  Define the office-supplies annotation list before collection; room location alone is not a class cue.
- A recognizable island is `kitchen_island`, not a duplicate table/cabinet. A door leaf and its opening are distinct
  entities and may overlap. Elevator entrances are not also `doorway`; elevator doors are excluded from `door_leaf`.
- Separate mirrors from transparent glass surfaces. Glass cups/bottles keep their object labels.
  `granular_debris` covers visible scattered particles such as cat litter; `hair_clump` does not promise single-hair detection.
- Review ambiguous appliance identities and wet/dry stains before training. Partial labels and ignore regions are
  unsupported; do not silently treat uncertain or unannotated targets as background. Review empty files as negatives.

## Separate geometry outputs

`cliff_edge`, `threshold`, `step_edge` and `floor_gap` are excluded from the 55 detector classes.
Their height, direction, width and confidence require geometric evidence and downstream rules.
The binary drivable-area model does not supply these outputs or the precise masks of the region classes above.

## Basis

The project taxonomy adapts [COCO](https://github.com/ultralytics/ultralytics/blob/922563eab42b778445e55a2edbabb6974eed8e4b/ultralytics/cfg/datasets/coco.yaml),
[Open Images](https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv) and
[LVIS](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/lvis_v1_categories.py)
to the ToC requirements. Category coverage does not establish accuracy at 640×352.
