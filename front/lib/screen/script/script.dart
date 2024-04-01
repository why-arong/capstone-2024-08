import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/widget/category_buttons.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';

class Script extends StatefulWidget {
  const Script({Key? key}) : super(key: key);

  @override
  State<Script> createState() => _ScriptState();
}

class _ScriptState extends State<Script> {
  String selectedCategoryValue = texts.category[0];

  AppBar scriptAppBar() {
    return AppBar(
     //backgroundColor: colors.bgrDarkColor,
     leading: Row(
      children: [
        TextButton(
          child: const Text('News'),
          onPressed: () { },
        ),
        TextButton(
          child: const Text('User'),
          onPressed: () { },
        )
      ]
    ),
    );
  }

  void _handleCategorySelected(String value) {
    setState(() {
      selectedCategoryValue = value;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: scriptAppBar(),
        body: Column(
          children: [
            CategoryButtons(onCategorySelected: (String value) {  },),
            SingleChildScrollView(

          )
        ])
    );
  }
}
