import 'package:capstone/widget/bottom_navigation.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      title: 'Capstone',
      initialRoute: '/bottom_navigation',
      getPages: [
        GetPage(
          name: '/bottom_navigation', 
          page: () => const BottomNavBar()
        )
      ],
      debugShowCheckedModeBanner: false,
    );
  }
}