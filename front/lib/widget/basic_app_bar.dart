import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:capstone/constants/color.dart' as colors;

AppBar basicAppBar({String title = '', Color backgroundColor = colors.bgrDarkColor}) {
    return AppBar(
      backgroundColor: backgroundColor,
      elevation: 0,
      title: Text(
        title,
        textAlign: TextAlign.center,
        semanticsLabel: title,
          style: TextStyle(
            color: backgroundColor == colors.bgrDarkColor ? colors.blockColor : colors.textColor,
            fontSize: 20,
            fontWeight: FontWeight.w700
          ),
        ),
      leading: IconButton(
        icon: const Icon(Icons.keyboard_backspace_rounded),
        onPressed: () => Get.back()),
    );
  }