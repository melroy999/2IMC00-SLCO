/**
 */
package nqc.provider;


import java.util.Collection;
import java.util.List;

import nqc.BuiltInUnaryValueFunctionEnum;
import nqc.NqcFactory;
import nqc.NqcPackage;
import nqc.UnaryBuiltInValueFunctionCall;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.ComposeableAdapterFactory;
import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.ViewerNotification;

/**
 * This is the item provider adapter for a {@link nqc.UnaryBuiltInValueFunctionCall} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class UnaryBuiltInValueFunctionCallItemProvider
	extends BuiltInValueFunctionCallItemProvider
	implements
		IEditingDomainItemProvider,
		IStructuredItemContentProvider,
		ITreeItemContentProvider,
		IItemLabelProvider,
		IItemPropertySource {
	/**
	 * This constructs an instance from a factory and a notifier.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public UnaryBuiltInValueFunctionCallItemProvider(AdapterFactory adapterFactory) {
		super(adapterFactory);
	}

	/**
	 * This returns the property descriptors for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public List<IItemPropertyDescriptor> getPropertyDescriptors(Object object) {
		if (itemPropertyDescriptors == null) {
			super.getPropertyDescriptors(object);

			addUnaryBuiltInValueFunctionPropertyDescriptor(object);
		}
		return itemPropertyDescriptors;
	}

	/**
	 * This adds a property descriptor for the Unary Built In Value Function feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addUnaryBuiltInValueFunctionPropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_UnaryBuiltInValueFunctionCall_UnaryBuiltInValueFunction_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_UnaryBuiltInValueFunctionCall_UnaryBuiltInValueFunction_feature", "_UI_UnaryBuiltInValueFunctionCall_type"),
				 NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_UnaryBuiltInValueFunction(),
				 true,
				 false,
				 false,
				 ItemPropertyDescriptor.GENERIC_VALUE_IMAGE,
				 null,
				 null));
	}

	/**
	 * This specifies how to implement {@link #getChildren} and is used to deduce an appropriate feature for an
	 * {@link org.eclipse.emf.edit.command.AddCommand}, {@link org.eclipse.emf.edit.command.RemoveCommand} or
	 * {@link org.eclipse.emf.edit.command.MoveCommand} in {@link #createCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Collection<? extends EStructuralFeature> getChildrenFeatures(Object object) {
		if (childrenFeatures == null) {
			super.getChildrenFeatures(object);
			childrenFeatures.add(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter());
		}
		return childrenFeatures;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EStructuralFeature getChildFeature(Object object, Object child) {
		// Check the type of the specified child object and return the proper feature to use for
		// adding (see {@link AddCommand}) it as a child.

		return super.getChildFeature(object, child);
	}

	/**
	 * This returns UnaryBuiltInValueFunctionCall.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/UnaryBuiltInValueFunctionCall"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		BuiltInUnaryValueFunctionEnum labelValue = ((UnaryBuiltInValueFunctionCall)object).getUnaryBuiltInValueFunction();
		String label = labelValue == null ? null : labelValue.toString();
		return label == null || label.length() == 0 ?
			getString("_UI_UnaryBuiltInValueFunctionCall_type") :
			getString("_UI_UnaryBuiltInValueFunctionCall_type") + " " + label;
	}

	/**
	 * This handles model notifications by calling {@link #updateChildren} to update any cached
	 * children and by creating a viewer notification, which it passes to {@link #fireNotifyChanged}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void notifyChanged(Notification notification) {
		updateChildren(notification);

		switch (notification.getFeatureID(UnaryBuiltInValueFunctionCall.class)) {
			case NqcPackage.UNARY_BUILT_IN_VALUE_FUNCTION_CALL__UNARY_BUILT_IN_VALUE_FUNCTION:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), false, true));
				return;
			case NqcPackage.UNARY_BUILT_IN_VALUE_FUNCTION_CALL__PARAMETER:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), true, false));
				return;
		}
		super.notifyChanged(notification);
	}

	/**
	 * This adds {@link org.eclipse.emf.edit.command.CommandParameter}s describing the children
	 * that can be created under this object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected void collectNewChildDescriptors(Collection<Object> newChildDescriptors, Object object) {
		super.collectNewChildDescriptors(newChildDescriptors, object);

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createAcquireConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createVariableExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createArrayExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createBinaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createBinaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createBooleanConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createDirectionConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createDisplayModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createEventTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createIntegerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createNullaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createOutputModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createOutputPortNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createSensorConfigConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createSensorModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createSensorNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createSensorTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createSerialBaudConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createSerialBiphaseConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createSerialChecksumConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createSerialChannelConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createSerialCommConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createSerialPacketConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createSoundConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createTernaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createTxPowerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createUnaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getUnaryBuiltInValueFunctionCall_Parameter(),
				 NqcFactory.eINSTANCE.createUnaryExpression()));
	}

}
